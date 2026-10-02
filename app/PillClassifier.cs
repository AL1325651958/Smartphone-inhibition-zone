using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Antibacterial_zone
{
    /// <summary>
    /// 药片缩写分类推理类，对应 class.onnx（药片CNN 的 DiskNet-small @224 分类器）。
    ///
    /// 与训练/评估管线严格对齐（药片CNN/train_augmented.py 的 FolderSet、
    /// ensemble_eval.py 的 predict）：
    ///   数据集里的图是**原始灰度裁剪**（未做 CLAHE 等增强），因此推理端只做
    ///     彩色裁剪 → 灰度(BT.601) → PIL 式 BICUBIC 缩放到 224×224
    ///     → 复制成 3 通道、数值 0..255（(x/255−0.5)/0.25 已烘焙进 ONNX）
    ///   5 视图 TTA：原图、水平翻转、垂直翻转、−10° 与 +10° 旋转（fill = 128）
    ///   多视图先对 **logits** 求平均再 softmax（与 ensemble_eval.py 一致）
    ///
    /// ONNX 接口（由 药片CNN/export_onnx_disknet.py 导出）：
    ///   输入  image  : float32 [N, 3, 224, 224]，0..255
    ///   输出  logits : float32 [N, 7]（未过 softmax）
    ///         probs  : float32 [N, 7]（softmax，供调试）
    ///   类别顺序固定为 ClassNames
    /// </summary>
    public class PillClassifier : IDisposable
    {
        // ── 配置（与训练/评估一致，勿随意改）──────────────────
        private const int InputSize = 224;
        private const int TtaViews = 5;

        public static readonly string[] ClassNames =
            { "CRO", "DA", "E", "LEV", "LZD", "P", "VA" };

        private readonly InferenceSession _session;
        private readonly string _logitsName;

        // ── 构造 / 释放 ───────────────────────────────────────
        public PillClassifier(byte[] modelBytes)
        {
            var opts = new SessionOptions();
            try { opts.AppendExecutionProvider_CUDA(); } catch { }
            _session = new InferenceSession(modelBytes, opts);

            _logitsName = _session.OutputMetadata.Keys.FirstOrDefault(
                k => k.Contains("logit", StringComparison.OrdinalIgnoreCase))
                ?? _session.OutputMetadata.Keys.First();
        }

        public void Dispose() => _session?.Dispose();

        // ══════════════════════════════════════════════════════
        //  对外接口：原始裁剪彩色图 → 识别 + 返回显示用灰度图
        // ══════════════════════════════════════════════════════
        public (string label, float confidence, SKBitmap displayBmp) ClassifyWithPreview(SKBitmap pillRaw)
        {
            if (pillRaw == null || pillRaw.IsNull) return ("?", 0f, null);
            try
            {
                int w = pillRaw.Width, h = pillRaw.Height;
                byte[] gray = ToGray(pillRaw, w, h);
                var (label, conf) = RunTta(gray, w, h);
                var displayBmp = GrayToBitmap(gray, w, h);

                System.Diagnostics.Debug.WriteLine(
                    $"[PillClassifier] {w}x{h} -> {label} {conf * 100f:F1}%");
                return (label, conf, displayBmp);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"[PillClassifier.ClassifyWithPreview] {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
                return ("Err", 0f, null);
            }
        }

        /// <summary>输入原始裁剪彩色图，只返回类别与置信度。</summary>
        public (string label, float confidence) Classify(SKBitmap pillCrop)
        {
            var (label, conf, bmp) = ClassifyWithPreview(pillCrop);
            bmp?.Dispose();
            return (label, conf);
        }

        /// <summary>输入已是灰度的裁剪图（原始裁剪尺寸），直接跑 TTA。</summary>
        public (string label, float confidence) ClassifyPreprocessed(SKBitmap grayCrop)
        {
            if (grayCrop == null || grayCrop.IsNull) return ("?", 0f);
            try
            {
                int w = grayCrop.Width, h = grayCrop.Height;
                var buf = new byte[w * h];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        buf[y * w + x] = grayCrop.GetPixel(x, y).Red;
                return RunTta(buf, w, h);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"[PillClassifier.ClassifyPreprocessed] {ex.GetType().Name}: {ex.Message}");
                return ("Err", 0f);
            }
        }

        /// <summary>
        /// 预处理原始图 → 灰度图（原始尺寸）。本模型的数据集是原始灰度裁剪，
        /// **不做** CLAHE/锐化/Gamma；保留此方法只是为了与旧调用点兼容。
        /// </summary>
        public static SKBitmap PreprocessToBitmap(SKBitmap src)
            => GrayToBitmap(ToGray(src, src.Width, src.Height), src.Width, src.Height);

        // ══════════════════════════════════════════════════════
        //  推理：灰度 → 224 → 5 视图 → logits 平均 → softmax
        // ══════════════════════════════════════════════════════
        private (string label, float confidence) RunTta(byte[] gray, int w, int h)
        {
            // 与数据集一致：先把裁剪图缩放到 224（数据集里就是 224 的正方形灰度图）
            byte[] b224 = ResamplePil(gray, w, h, InputSize, InputSize);

            var views = new List<byte[]>(TtaViews) { b224 };
            if (TtaViews > 1)
            {
                views.Add(FlipHorizontal(b224, InputSize, InputSize));
                views.Add(FlipVertical(b224, InputSize, InputSize));
            }
            if (TtaViews >= 5)
            {
                // 与 ensemble_eval.py 一致：-10° 与 +10°，fill = 128
                views.Add(RotateBicubic(b224, InputSize, InputSize, -10.0, 128));
                views.Add(RotateBicubic(b224, InputSize, InputSize, +10.0, 128));
            }

            var logits = RunBatch(views);
            int n = ClassNames.Length;
            var mean = new float[n];
            for (int v = 0; v < views.Count; v++)
                for (int i = 0; i < n; i++)
                    mean[i] += logits[v * n + i];
            for (int i = 0; i < n; i++) mean[i] /= views.Count;

            var probs = Softmax(mean);
            int top = 0;
            for (int i = 1; i < probs.Length; i++) if (probs[i] > probs[top]) top = i;
            string label = top < ClassNames.Length ? ClassNames[top] : top.ToString();

            System.Diagnostics.Debug.WriteLine(
                $"[PillClassifier] views={views.Count} -> {label} {probs[top] * 100f:F1}%");
            return (label, probs[top]);
        }

        /// <summary>一次前向跑完所有 TTA 视图，返回 [views × 7] 的 logits。</summary>
        private float[] RunBatch(List<byte[]> views)
        {
            int n = views.Count;
            var tensor = new DenseTensor<float>(new[] { n, 3, InputSize, InputSize });
            for (int v = 0; v < n; v++)
            {
                var buf = views[v];
                for (int y = 0; y < InputSize; y++)
                    for (int x = 0; x < InputSize; x++)
                    {
                        float val = buf[y * InputSize + x];   // 0..255，归一化在图内
                        tensor[v, 0, y, x] = val;
                        tensor[v, 1, y, x] = val;
                        tensor[v, 2, y, x] = val;
                    }
            }

            var inputName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            });

            NamedOnnxValue chosen = null;
            foreach (var r in results)
                if (r.Name == _logitsName) { chosen = r; break; }
            chosen ??= results.First();

            var t = chosen.AsTensor<float>();
            var outv = new float[t.Length];
            for (int i = 0; i < outv.Length; i++) outv[i] = t.GetValue(i);
            return outv;
        }

        private static float[] Softmax(float[] logits)
        {
            float max = logits[0];
            for (int i = 1; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
            double sum = 0;
            var exps = new double[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                exps[i] = Math.Exp(logits[i] - max);
                sum += exps[i];
            }
            var probs = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++) probs[i] = (float)(exps[i] / sum);
            return probs;
        }

        // ══════════════════════════════════════════════════════
        //  基础图像操作
        // ══════════════════════════════════════════════════════
        private static byte[] ToGray(SKBitmap bmp, int w, int h)
        {
            var buf = new byte[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    var p = bmp.GetPixel(x, y);
                    buf[y * w + x] =
                        (byte)(0.299f * p.Red + 0.587f * p.Green + 0.114f * p.Blue + 0.5f);
                }
            return buf;
        }

        private static SKBitmap GrayToBitmap(byte[] gray, int w, int h)
        {
            var bmp = new SKBitmap(w, h);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    byte v = gray[y * w + x];
                    bmp.SetPixel(x, y, new SKColor(v, v, v, 255));
                }
            return bmp;
        }

        private static byte[] FlipHorizontal(byte[] src, int w, int h)
        {
            var dst = new byte[src.Length];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[y * w + x] = src[y * w + (w - 1 - x)];
            return dst;
        }

        private static byte[] FlipVertical(byte[] src, int w, int h)
        {
            var dst = new byte[src.Length];
            for (int y = 0; y < h; y++)
                Array.Copy(src, (h - 1 - y) * w, dst, y * w, w);
            return dst;
        }

        // ══════════════════════════════════════════════════════
        //  PIL 风格可分离重采样（BICUBIC，向下缩放时按 scale 扩大支撑 → 抗锯齿）
        //  与 Pillow Image.resize(..., Image.BICUBIC) 对齐
        // ══════════════════════════════════════════════════════
        private static byte[] ResamplePil(byte[] src, int sw, int sh, int dw, int dh)
        {
            if (sw == dw && sh == dh) return src;
            byte[] tmp = ResampleHorizontal(src, sw, sh, dw);
            return ResampleVertical(tmp, dw, sh, dh);
        }

        private static byte[] ResampleHorizontal(byte[] src, int sw, int sh, int dw)
        {
            var dst = new byte[dw * sh];
            double scale = (double)sw / dw;
            double filterscale = Math.Max(1.0, scale);
            double support = 2.0 * filterscale;

            for (int x = 0; x < dw; x++)
            {
                double center = (x + 0.5) * scale;
                int xmin = (int)Math.Floor(center - support + 0.5);
                int xmax = (int)Math.Floor(center + support + 0.5);
                if (xmin < 0) xmin = 0;
                if (xmax > sw) xmax = sw;

                int taps = xmax - xmin;
                var wts = new double[taps];
                double wsum = 0;
                for (int i = 0; i < taps; i++)
                {
                    double wgt = CubicKernel((xmin + i - center + 0.5) / filterscale);
                    wts[i] = wgt;
                    wsum += wgt;
                }
                if (wsum == 0) { wts[0] = 1; wsum = 1; }

                for (int y = 0; y < sh; y++)
                {
                    int row = y * sw;
                    double acc = 0;
                    for (int i = 0; i < taps; i++) acc += wts[i] * src[row + xmin + i];
                    dst[y * dw + x] = (byte)Math.Clamp((int)(acc / wsum + 0.5), 0, 255);
                }
            }
            return dst;
        }

        private static byte[] ResampleVertical(byte[] src, int w, int sh, int dh)
        {
            var dst = new byte[w * dh];
            double scale = (double)sh / dh;
            double filterscale = Math.Max(1.0, scale);
            double support = 2.0 * filterscale;

            for (int y = 0; y < dh; y++)
            {
                double center = (y + 0.5) * scale;
                int ymin = (int)Math.Floor(center - support + 0.5);
                int ymax = (int)Math.Floor(center + support + 0.5);
                if (ymin < 0) ymin = 0;
                if (ymax > sh) ymax = sh;

                int taps = ymax - ymin;
                var wts = new double[taps];
                double wsum = 0;
                for (int i = 0; i < taps; i++)
                {
                    double wgt = CubicKernel((ymin + i - center + 0.5) / filterscale);
                    wts[i] = wgt;
                    wsum += wgt;
                }
                if (wsum == 0) { wts[0] = 1; wsum = 1; }

                for (int x = 0; x < w; x++)
                {
                    double acc = 0;
                    for (int i = 0; i < taps; i++) acc += wts[i] * src[(ymin + i) * w + x];
                    dst[y * w + x] = (byte)Math.Clamp((int)(acc / wsum + 0.5), 0, 255);
                }
            }
            return dst;
        }

        /// <summary>Catmull-Rom 三次核（Pillow bicubic，a = -0.5）。</summary>
        private static double CubicKernel(double x)
        {
            x = Math.Abs(x);
            const double a = -0.5;
            if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;
            if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
            return 0.0;
        }

        /// <summary>
        /// 旋转（Pillow 语义：正角度 = 逆时针），中心取 (w/2, h/2) 与 PIL rotate 一致，
        /// 采样用双三次核，落点超出原图用 fill 值（PIL 的 fillcolor）。
        /// </summary>
        private static byte[] RotateBicubic(byte[] src, int w, int h, double degrees, byte fill)
        {
            var dst = new byte[w * h];
            double rad = degrees * Math.PI / 180.0;
            double cos = Math.Cos(rad), sin = Math.Sin(rad);
            double cx = w / 2.0, cy = h / 2.0;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                    double sx = cos * dx + sin * dy + cx - 0.5;
                    double sy = -sin * dx + cos * dy + cy - 0.5;
                    dst[y * w + x] = SampleBicubic(src, w, h, sx, sy, fill);
                }
            }
            return dst;
        }

        private static byte SampleBicubic(byte[] src, int w, int h, double x, double y, byte fill)
        {
            if (x < -1.0 || y < -1.0 || x > w || y > h) return fill;

            int x0 = (int)Math.Floor(x);
            int y0 = (int)Math.Floor(y);
            double acc = 0, wsum = 0;
            for (int j = -1; j <= 2; j++)
            {
                int sy = y0 + j;
                double wy = CubicKernel(y - sy);
                if (wy == 0) continue;
                for (int i = -1; i <= 2; i++)
                {
                    int sx = x0 + i;
                    double wx = CubicKernel(x - sx);
                    if (wx == 0) continue;
                    double wgt = wx * wy;
                    if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
                    acc += wgt * src[sy * w + sx];
                    wsum += wgt;
                }
            }
            if (wsum == 0) return fill;
            return (byte)Math.Clamp((int)(acc / wsum + 0.5), 0, 255);
        }
    }
}
