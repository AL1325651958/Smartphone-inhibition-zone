using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Antibacterial_zone
{
    /// <summary>
    /// 自研语义分割模型（semseg：ResNet18 + U-Net）的 ONNX 封装，替换原来的 YOLOv8n-seg。
    ///
    /// ONNX 接口约定（由 semseg/export_onnx.py 导出）：
    ///   输入  [1, 3, 640, 640] float32，RGB，数值范围 0..255
    ///         （ImageNet 均值/方差的归一化已烘焙进计算图）
    ///   输出  [1, 2, 640, 640] float32，逐通道 sigmoid 概率
    ///         通道 0 = Area（抑菌圈），通道 1 = Yaoping（药片）
    ///
    /// 与训练一致的预处理（semseg/dataset.py）：
    ///   直接双线性 resize 到 640×640（**不做 letterbox，不补灰边**），BGR → RGB。
    ///
    /// 实例化（模型只输出语义 2 通道，需要在这里拆分实例）：
    ///   药片  = 药片通道的 8 连通域（面积 ≥ MinArea）
    ///   抑菌圈 = 以药片质心为种子的测地分区（多源 BFS），相邻融合的圈在此被切开，
    ///           每个药片最多得到一块圈区域；没有分到区域的药片 has_zone = false。
    /// </summary>
    public class ZoneSegModel : IDisposable
    {
        public const int InputSize = 640;
        /// <summary>两通道共用的概率阈值（semseg/config.py 的 THRESH）。</summary>
        public const float Threshold = 0.45f;
        /// <summary>640 尺度下小于该面积的目标丢弃（semseg/infer.py 的 min_area）。</summary>
        public const int MinArea = 30;
        /// <summary>测试时增强：0/90/180/270 旋转 + 水平翻转共 5 视图取平均（与交付配置一致）。</summary>
        public static bool UseTta = true;
        /// <summary>种子回退搜索半径（像素，640 尺度）：药片质心不在圈掩膜内时向外找最近圈像素。</summary>
        private const int SeedSearchRadius = 32;

        private const int Zone = 0;   // Area
        private const int Disk = 1;   // Yaoping

        private static readonly string[] ClassNames = { "Area", "Yaoping" };

        private static readonly SKColor ColorArea = new SKColor(29, 158, 117);   // 绿：抑菌区域
        private static readonly SKColor ColorYaoping = new SKColor(24, 95, 165); // 蓝：药片

        private readonly InferenceSession _session;

        // 预处理时记录，后处理还原坐标用
        private float _scaleX = 1f;   // 640 / 原图宽
        private float _scaleY = 1f;   // 640 / 原图高
        private int _origW, _origH;

        public ZoneSegModel(byte[] modelBytes)
        {
            var options = new SessionOptions();
            try { options.AppendExecutionProvider_CUDA(); } catch { }
            _session = new InferenceSession(modelBytes, options);
        }

        public void Dispose() => _session?.Dispose();

        // ══════════════════════════════════════════
        //  预处理：双线性 resize 到 640×640（无灰边、各向异性缩放）
        // ══════════════════════════════════════════
        public SKBitmap PreprocessImage(SKBitmap original)
        {
            _origW = original.Width;
            _origH = original.Height;
            _scaleX = (float)InputSize / _origW;
            _scaleY = (float)InputSize / _origH;
            return ResizeBilinear(original, InputSize, InputSize);
        }

        // ══════════════════════════════════════════
        //  推理入口
        // ══════════════════════════════════════════
        public List<SegInstance> Predict(SKBitmap preprocessed)
        {
            float[] x = ToNchwRgb(preprocessed);      // [3,640,640]，0..255
            float[] probs = RunWithTta(x);            // [2,640,640]
            return BuildInstances(probs);
        }

        // ── SKBitmap → NCHW RGB float（0..255，归一化在图内）──
        private static float[] ToNchwRgb(SKBitmap bmp)
        {
            int s = InputSize;
            var x = new float[3 * s * s];
            int plane = s * s;
            for (int y = 0; y < s; y++)
            {
                for (int px = 0; px < s; px++)
                {
                    var c = bmp.GetPixel(px, y);
                    int i = y * s + px;
                    x[i] = c.Red;
                    x[plane + i] = c.Green;
                    x[2 * plane + i] = c.Blue;
                }
            }
            return x;
        }

        // ── 单次前向：输出 [2,640,640] 概率 ──
        private float[] RunOnce(float[] x)
        {
            int s = InputSize;
            var tensor = new DenseTensor<float>(x, new[] { 1, 3, s, s });
            var inputName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(
                new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) });

            var output = results.First().AsTensor<float>();
            int total = 2 * s * s;
            var probs = new float[total];
            for (int i = 0; i < total; i++) probs[i] = output.GetValue(i);
            return probs;
        }

        // ── 5 视图 TTA（与 semseg/infer.py 的 _forward 完全一致）──
        private float[] RunWithTta(float[] x)
        {
            int s = InputSize;

            // 单次
            float[] acc = RunOnce(x);

            if (!UseTta) return acc;

            // 旋转 90/180/270：输入 rot90(k)，输出反向转回
            for (int k = 1; k <= 3; k++)
            {
                float[] xr = Rot90Plane(x, 3, s, k);
                float[] pr = Rot90BackPlane(RunOnce(xr), 2, s, k);
                for (int i = 0; i < acc.Length; i++) acc[i] += pr[i];
            }

            // 水平翻转
            float[] xf = FlipPlaneH(x, 3, s);
            float[] pf = FlipPlaneH(RunOnce(xf), 2, s);
            for (int i = 0; i < acc.Length; i++) acc[i] += pf[i];

            for (int i = 0; i < acc.Length; i++) acc[i] /= 5f;
            return acc;
        }

        // ══════════════════════════════════════════
        //  实例化：药片连通域 + 圈测地分区
        // ══════════════════════════════════════════
        private List<SegInstance> BuildInstances(float[] probs)
        {
            int s = InputSize;
            int plane = s * s;

            bool[] diskBin = new bool[plane];
            bool[] zoneBin = new bool[plane];
            for (int i = 0; i < plane; i++)
            {
                diskBin[i] = probs[plane + i] > Threshold;   // 通道 1 = 药片
                zoneBin[i] = probs[i] > Threshold;           // 通道 0 = 抑菌圈
            }

            // ── 1) 药片：8 连通域 ──
            var diskComps = ConnectedComponents8(diskBin, s, s, MinArea, probs, plane);
            var disks = new List<SegInstance>();
            for (int k = 0; k < diskComps.Count; k++)
            {
                var c = diskComps[k];
                disks.Add(new SegInstance
                {
                    ClassId = Disk,
                    ClassName = ClassNames[Disk],
                    Confidence = c.MeanProb,
                    MaskW = s,
                    MaskH = s,
                    Mask = c.Mask,
                    X = c.MinX,
                    Y = c.MinY,
                    Width = c.MaxX - c.MinX + 1,
                    Height = c.MaxY - c.MinY + 1,
                    DiskIndex = k,
                    HasZone = false,
                });
            }

            // ── 2) 抑菌圈：以药片质心为种子的多源 BFS 分区 ──
            int[] owner = new int[plane];
            for (int i = 0; i < plane; i++) owner[i] = -1;

            var queue = new int[plane];
            int head = 0, tail = 0;

            for (int k = 0; k < diskComps.Count; k++)
            {
                var c = diskComps[k];
                int sx = (int)Math.Round(c.CentroidX);
                int sy = (int)Math.Round(c.CentroidY);
                if (!FindSeed(zoneBin, s, ref sx, ref sy)) continue;   // 该药片附近没有圈像素

                int idx = sy * s + sx;
                if (owner[idx] < 0)
                {
                    owner[idx] = k;
                    queue[tail++] = idx;
                }
            }

            // 8 邻域 BFS
            while (head < tail)
            {
                int cur = queue[head++];
                int cx = cur % s, cy = cur / s;
                int own = owner[cur];
                for (int dy = -1; dy <= 1; dy++)
                {
                    int ny = cy + dy;
                    if (ny < 0 || ny >= s) continue;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int nx = cx + dx;
                        if (nx < 0 || nx >= s) continue;
                        int ni = ny * s + nx;
                        if (!zoneBin[ni] || owner[ni] >= 0) continue;
                        owner[ni] = own;
                        queue[tail++] = ni;
                    }
                }
            }

            // 统计每个药片分到的圈像素
            var counts = new int[diskComps.Count];
            var sumProb = new double[diskComps.Count];
            var minX = new int[diskComps.Count];
            var minY = new int[diskComps.Count];
            var maxX = new int[diskComps.Count];
            var maxY = new int[diskComps.Count];
            for (int k = 0; k < diskComps.Count; k++)
            {
                minX[k] = int.MaxValue; minY[k] = int.MaxValue;
                maxX[k] = -1; maxY[k] = -1;
            }
            int orphanZone = 0;
            for (int i = 0; i < plane; i++)
            {
                if (!zoneBin[i]) continue;
                int o = owner[i];
                if (o < 0) { orphanZone++; continue; }
                counts[o]++;
                sumProb[o] += probs[i];
                int px = i % s, py = i / s;
                if (px < minX[o]) minX[o] = px;
                if (py < minY[o]) minY[o] = py;
                if (px > maxX[o]) maxX[o] = px;
                if (py > maxY[o]) maxY[o] = py;
            }

            var zones = new List<SegInstance>();
            for (int k = 0; k < diskComps.Count; k++)
            {
                if (counts[k] < MinArea) continue;
                var mask = new byte[plane];
                for (int i = 0; i < plane; i++)
                    if (owner[i] == k) mask[i] = 1;

                zones.Add(new SegInstance
                {
                    ClassId = Zone,
                    ClassName = ClassNames[Zone],
                    Confidence = (float)(sumProb[k] / counts[k]),
                    MaskW = s,
                    MaskH = s,
                    Mask = mask,
                    X = minX[k],
                    Y = minY[k],
                    Width = maxX[k] - minX[k] + 1,
                    Height = maxY[k] - minY[k] + 1,
                    DiskIndex = k,
                    HasZone = true,
                    ZoneAreaPx = counts[k],
                });
                disks[k].HasZone = true;
            }

            LastOrphanZonePixels = orphanZone;

            var result = new List<SegInstance>(disks.Count + zones.Count);
            result.AddRange(disks);
            result.AddRange(zones);
            return result;
        }

        /// <summary>孤立（不与任何药片连通）的圈像素数，用于诊断。</summary>
        public int LastOrphanZonePixels { get; private set; }

        // 种子定位：质心在圈掩膜内直接用；否则在半径内找最近的圈像素
        private static bool FindSeed(bool[] zoneBin, int s, ref int sx, ref int sy)
        {
            sx = Math.Clamp(sx, 0, s - 1);
            sy = Math.Clamp(sy, 0, s - 1);
            if (zoneBin[sy * s + sx]) return true;

            int best = int.MaxValue, bx = -1, by = -1;
            int r = SeedSearchRadius;
            for (int dy = -r; dy <= r; dy++)
            {
                int ny = sy + dy;
                if (ny < 0 || ny >= s) continue;
                for (int dx = -r; dx <= r; dx++)
                {
                    int nx = sx + dx;
                    if (nx < 0 || nx >= s) continue;
                    if (!zoneBin[ny * s + nx]) continue;
                    int d = dx * dx + dy * dy;
                    if (d < best) { best = d; bx = nx; by = ny; }
                }
            }
            if (bx < 0) return false;
            sx = bx; sy = by;
            return true;
        }

        // ── 8 连通域（面积过滤，返回掩膜/包围盒/质心/平均概率）──
        private sealed class Component
        {
            public byte[] Mask;
            public int Area;
            public int MinX, MinY, MaxX, MaxY;
            public float CentroidX, CentroidY;
            public float MeanProb;
        }

        private static List<Component> ConnectedComponents8(
            bool[] bin, int w, int h, int minArea, float[] probs, int probOffset)
        {
            var comps = new List<Component>();
            int[] label = new int[w * h];
            var stack = new int[w * h];

            for (int start = 0; start < label.Length; start++)
            {
                if (!bin[start] || label[start] != 0) continue;

                int sp = 0;
                stack[sp++] = start;
                label[start] = 1;
                var pixels = new List<int>();

                while (sp > 0)
                {
                    int cur = stack[--sp];
                    pixels.Add(cur);
                    int cx = cur % w, cy = cur / w;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int ny = cy + dy;
                        if (ny < 0 || ny >= h) continue;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0) continue;
                            int nx = cx + dx;
                            if (nx < 0 || nx >= w) continue;
                            int ni = ny * w + nx;
                            if (!bin[ni] || label[ni] != 0) continue;
                            label[ni] = 1;
                            stack[sp++] = ni;
                        }
                    }
                }

                if (pixels.Count < minArea) { label[start] = 2; continue; }

                var mask = new byte[w * h];
                int minX = int.MaxValue, minY = int.MaxValue, maxX = -1, maxY = -1;
                double sx = 0, sy = 0, sp2 = 0;
                foreach (int i in pixels)
                {
                    mask[i] = 1;
                    int px = i % w, py = i / w;
                    if (px < minX) minX = px;
                    if (py < minY) minY = py;
                    if (px > maxX) maxX = px;
                    if (py > maxY) maxY = py;
                    sx += px; sy += py;
                    sp2 += probs[probOffset + i];
                }
                comps.Add(new Component
                {
                    Mask = mask,
                    Area = pixels.Count,
                    MinX = minX, MinY = minY, MaxX = maxX, MaxY = maxY,
                    CentroidX = (float)(sx / pixels.Count),
                    CentroidY = (float)(sy / pixels.Count),
                    MeanProb = (float)(sp2 / pixels.Count),
                });
            }
            return comps;
        }

        // ══════════════════════════════════════════
        //  掩膜坐标（640）→ 原图坐标（各向异性）
        // ══════════════════════════════════════════
        public SKRect ToOriginalRect(SegInstance inst)
        {
            float x = inst.X / _scaleX;
            float y = inst.Y / _scaleY;
            float w = inst.Width / _scaleX;
            float h = inst.Height / _scaleY;
            x = Math.Max(0, Math.Min(x, _origW));
            y = Math.Max(0, Math.Min(y, _origH));
            w = Math.Min(w, _origW - x);
            h = Math.Min(h, _origH - y);
            return new SKRect(x, y, x + w, y + h);
        }

        /// <summary>实例在原图中的面积（px²）：直接用掩膜像素数换算，比轮廓 shoelace 稳。</summary>
        public float AreaInOriginal(SegInstance inst)
        {
            int count = 0;
            var m = inst.Mask;
            for (int i = 0; i < m.Length; i++) if (m[i] != 0) count++;
            return count / (_scaleX * _scaleY);
        }

        /// <summary>等效圆直径（原图 px）：d = 2√(A/π)。</summary>
        public float EquivalentDiameter(SegInstance inst)
        {
            float a = AreaInOriginal(inst);
            return a > 0 ? 2f * MathF.Sqrt(a / MathF.PI) : 0f;
        }

        /// <summary>
        /// 逐行扫描提取轮廓（原图坐标）。掩膜来自语义模型，实例已由分区保证单连通，
        /// 逐行左右边界即是可用轮廓（与旧版 YOLO 掩膜的取轮廓方式一致）。
        /// </summary>
        public List<SKPoint> GetContourPointsInOriginal(SegInstance inst)
        {
            var contour = new List<SKPoint>();
            if (inst?.Mask == null) return contour;

            int w = inst.MaskW;
            int x0 = (int)inst.X, y0 = (int)inst.Y;
            int x1 = Math.Min(w - 1, (int)(inst.X + inst.Width - 1));
            int y1 = Math.Min(inst.MaskH - 1, (int)(inst.Y + inst.Height - 1));

            var left = new List<SKPoint>();
            var right = new List<SKPoint>();

            for (int y = y0; y <= y1; y++)
            {
                int lx = -1, rx = -1;
                for (int x = x0; x <= x1; x++)
                {
                    if (inst.Mask[y * w + x] == 0) continue;
                    if (lx < 0) lx = x;
                    rx = x;
                }
                if (lx < 0) continue;

                float oy = (y + 0.5f) / _scaleY;
                left.Add(new SKPoint((lx + 0.5f) / _scaleX, oy));
                right.Add(new SKPoint((rx + 0.5f) / _scaleX, oy));
            }

            if (left.Count == 0) return contour;
            contour.AddRange(left);
            right.Reverse();
            contour.AddRange(right);
            return contour;
        }

        /// <summary>640×640 的二值掩膜位图（半透明填充用）。</summary>
        public SKBitmap GetMaskBitmap(SegInstance inst)
        {
            int w = inst.MaskW, h = inst.MaskH;
            var colors = new SKColor[w * h];
            var white = new SKColor(255, 255, 255, 255);
            var clear = new SKColor(0, 0, 0, 0);
            for (int i = 0; i < colors.Length; i++)
                colors[i] = inst.Mask[i] != 0 ? white : clear;

            var bmp = new SKBitmap(w, h);
            bmp.Pixels = colors;
            return bmp;
        }

        // ══════════════════════════════════════════
        //  可视化
        // ══════════════════════════════════════════
        public SKBitmap DrawPredictions(SKBitmap original, List<SegInstance> preds)
        {
            var output = new SKBitmap(original.Width, original.Height,
                                      original.ColorType, original.AlphaType);
            using (var initCanvas = new SKCanvas(output))
                initCanvas.DrawBitmap(original, 0, 0);

            using var canvas = new SKCanvas(output);
            var fullRect = new SKRect(0, 0, original.Width, original.Height);

            foreach (var pred in preds)
            {
                var color = pred.ClassId == Zone ? ColorArea : ColorYaoping;

                using (var maskBmp = GetMaskBitmap(pred))
                using (var fillPaint = new SKPaint
                {
                    // 用 SrcIn 混合把白色掩膜染成类别色（SkiaSharp 里 DrawBitmap 不吃 SKPaint.Color）
                    ColorFilter = SKColorFilter.CreateBlendMode(
                        color.WithAlpha(55), SKBlendMode.SrcIn),
                    IsAntialias = false,
                })
                {
                    // 掩膜覆盖整帧（预处理是整图 resize，无 padding）
                    canvas.DrawBitmap(maskBmp, fullRect, fillPaint);
                }

                var contourPts = GetContourPointsInOriginal(pred);
                if (contourPts.Count < 3) continue;

                var path = new SKPath();
                path.MoveTo(contourPts[0]);
                foreach (var pt in contourPts.Skip(1)) path.LineTo(pt);
                path.Close();

                using (var strokePaint = new SKPaint
                {
                    Color = color,
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 2.5f,
                    IsAntialias = true,
                })
                {
                    canvas.DrawPath(path, strokePaint);
                }

                float eqDiam = EquivalentDiameter(pred);
                var rect = ToOriginalRect(pred);
                string label = pred.ClassId == Zone
                    ? $"{pred.ClassName}  d={eqDiam:F0}px"
                    : $"{pred.ClassName} {pred.Confidence:P0}  d={eqDiam:F0}px";
                float textX = Math.Max(rect.Left, 4f);
                float textY = Math.Max(rect.Top - 6f, 22f);

                using var bgPaint = new SKPaint
                {
                    Color = color.WithAlpha(200),
                    Style = SKPaintStyle.Fill,
                    IsAntialias = true,
                };
                using var txtPaint = new SKPaint
                {
                    Color = SKColors.White,
                    TextSize = 18,
                    IsAntialias = true,
                    Typeface = SKTypeface.FromFamilyName("Arial",
                        SKFontStyleWeight.Bold, SKFontStyleWidth.Normal, SKFontStyleSlant.Upright),
                };
                float txtW = txtPaint.MeasureText(label);
                canvas.DrawRoundRect(
                    new SKRoundRect(
                        new SKRect(textX - 2, textY - 18, textX + txtW + 6, textY + 4), 4),
                    bgPaint);
                canvas.DrawText(label, textX + 2, textY, txtPaint);
            }

            return output;
        }

        // ══════════════════════════════════════════
        //  几何工具：双线性 resize / 张量旋转翻转
        // ══════════════════════════════════════════
        /// <summary>双线性缩放（坐标映射与 cv2.resize INTER_LINEAR 一致）。</summary>
        private static SKBitmap ResizeBilinear(SKBitmap src, int dw, int dh)
        {
            int sw = src.Width, sh = src.Height;
            var dst = new SKBitmap(dw, dh);
            float scX = (float)sw / dw;
            float scY = (float)sh / dh;

            for (int dy = 0; dy < dh; dy++)
            {
                float fy = (dy + 0.5f) * scY - 0.5f;
                int y0 = (int)MathF.Floor(fy);
                float ty = fy - y0;
                int y1 = Math.Min(y0 + 1, sh - 1);
                if (y0 < 0) y0 = 0;

                for (int dx = 0; dx < dw; dx++)
                {
                    float fx = (dx + 0.5f) * scX - 0.5f;
                    int x0 = (int)MathF.Floor(fx);
                    float tx = fx - x0;
                    int x1 = Math.Min(x0 + 1, sw - 1);
                    if (x0 < 0) x0 = 0;

                    var c00 = src.GetPixel(x0, y0);
                    var c10 = src.GetPixel(x1, y0);
                    var c01 = src.GetPixel(x0, y1);
                    var c11 = src.GetPixel(x1, y1);

                    byte Blend(byte a, byte b, byte c, byte d)
                    {
                        float v = a * (1 - tx) * (1 - ty) + b * tx * (1 - ty)
                                + c * (1 - tx) * ty + d * tx * ty;
                        return (byte)Math.Clamp((int)(v + 0.5f), 0, 255);
                    }

                    dst.SetPixel(dx, dy, new SKColor(
                        Blend(c00.Red, c10.Red, c01.Red, c11.Red),
                        Blend(c00.Green, c10.Green, c01.Green, c11.Green),
                        Blend(c00.Blue, c10.Blue, c01.Blue, c11.Blue),
                        255));
                }
            }
            return dst;
        }

        // np.rot90(A, k)（逆时针）作用于 (C,S,S) 的每个平面
        private static float[] Rot90Plane(float[] src, int c, int s, int times)
        {
            var cur = src;
            for (int t = 0; t < times; t++)
            {
                var dst = new float[cur.Length];
                int plane = s * s;
                for (int ch = 0; ch < c; ch++)
                {
                    int off = ch * plane;
                    for (int r = 0; r < s; r++)
                        for (int col = 0; col < s; col++)
                            dst[off + r * s + col] = cur[off + col * s + (s - 1 - r)];
                }
                cur = dst;
            }
            return cur;
        }

        // 逆运算（顺时针）
        private static float[] Rot90BackPlane(float[] src, int c, int s, int times)
        {
            var cur = src;
            for (int t = 0; t < times; t++)
            {
                var dst = new float[cur.Length];
                int plane = s * s;
                for (int ch = 0; ch < c; ch++)
                {
                    int off = ch * plane;
                    for (int r = 0; r < s; r++)
                        for (int col = 0; col < s; col++)
                            dst[off + r * s + col] = cur[off + (s - 1 - col) * s + r];
                }
                cur = dst;
            }
            return cur;
        }

        private static float[] FlipPlaneH(float[] src, int c, int s)
        {
            var dst = new float[src.Length];
            int plane = s * s;
            for (int ch = 0; ch < c; ch++)
            {
                int off = ch * plane;
                for (int r = 0; r < s; r++)
                    for (int col = 0; col < s; col++)
                        dst[off + r * s + col] = src[off + r * s + (s - 1 - col)];
            }
            return dst;
        }
    }

    // ══════════════════════════════════════════
    //  数据类
    // ══════════════════════════════════════════
    public class SegInstance
    {
        /// <summary>0 = Area（抑菌圈），1 = Yaoping（药片）。</summary>
        public int ClassId { get; set; }
        public string ClassName { get; set; }
        public float Confidence { get; set; }

        public int MaskW { get; set; }
        public int MaskH { get; set; }
        /// <summary>640×640 二值掩膜（0/1）。</summary>
        public byte[] Mask { get; set; }

        // 640 坐标系下的包围盒
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }

        /// <summary>药片：自身序号；抑菌圈：所属药片序号（测地分区结果）。</summary>
        public int DiskIndex { get; set; } = -1;
        /// <summary>仅药片实例：是否分到了抑菌圈区域。</summary>
        public bool HasZone { get; set; }
        /// <summary>仅抑菌圈实例：640 尺度下的像素数。</summary>
        public int ZoneAreaPx { get; set; }
    }
}
