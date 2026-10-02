using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Antibacterial_zone
{
    public class ImageProcessor
    {
        private readonly List<PillAntibacterialPair> _pairs = new();
        private int _currentIndex = -1;

        // 新增：保存“已在原图上标记几何中心和药片外接圆”的图
        private SKBitmap _markedOriginalImage;

        public int CurrentIndex => _currentIndex;
        public int TotalRegions => _pairs.Count;

        /// <summary>
        /// 返回已在原图上绘制了：
        /// 1) 所有药片中心点构成的最小外接圆
        /// 2) 该外接圆圆心（几何中心 P）
        /// 的图像
        /// </summary>
        public SKBitmap GetMarkedOriginalImage() => _markedOriginalImage;

        public void ProcessDetectedRegions(
            SKBitmap original,
            List<SegInstance> predictions,
            ZoneSegModel model,
            IProgress<double> progress = null,
            PillClassifier classifier = null)
        {
            Clear();

            var pills = predictions.Where(p => p.ClassId == 1).ToList();
            var areas = predictions.Where(p => p.ClassId == 0).ToList();
            int total = Math.Max(areas.Count, 1);
            int done = 0;

            var availablePills = new List<SegInstance>(pills);

            // ─────────────────────────────────────────────
            // 1) 计算所有药片的“中心点”
            //    药片中心优先用轮廓 bbox 中心；没有轮廓则退化到检测框中心
            // ─────────────────────────────────────────────
            var allPillCenters = new List<SKPoint>();

            foreach (var pill in pills)
            {
                var pillPts = model.GetContourPointsInOriginal(pill);
                if (pillPts != null && pillPts.Count >= 3)
                {
                    float minX = pillPts.Min(p => p.X);
                    float maxX = pillPts.Max(p => p.X);
                    float minY = pillPts.Min(p => p.Y);
                    float maxY = pillPts.Max(p => p.Y);

                    allPillCenters.Add(new SKPoint(
                        (minX + maxX) / 2f,
                        (minY + maxY) / 2f));
                }
                else
                {
                    var pr = model.ToOriginalRect(pill);
                    allPillCenters.Add(new SKPoint(pr.MidX, pr.MidY));
                }
            }

            // ─────────────────────────────────────────────
            // 2) 几何中心 P = 所有药片中心点构成集合的最小外接圆圆心
            //    同时得到这个外接圆半径
            // ─────────────────────────────────────────────
            SKPoint globalP = default;
            float globalPillGroupRadius = 0f;
            bool hasGlobalP = false;

            if (allPillCenters.Count > 0)
            {
                ComputeMinEnclosingCircle(allPillCenters, out globalP, out globalPillGroupRadius);
                hasGlobalP = true;
            }

            // ─────────────────────────────────────────────
            // 3) 在原图上标记几何中心 P，并画出“所有药片构成的外接圆”
            // ─────────────────────────────────────────────
            _markedOriginalImage = original.Copy();

            using (var canvas = new SKCanvas(_markedOriginalImage))
            {
                if (hasGlobalP)
                {
                    // 外接圆
                    canvas.DrawCircle(globalP, globalPillGroupRadius, new SKPaint
                    {
                        Color = new SKColor(255, 80, 80, 220),
                        Style = SKPaintStyle.Stroke,
                        StrokeWidth = 3f,
                        IsAntialias = true,
                        PathEffect = SKPathEffect.CreateDash(new[] { 10f, 6f }, 0)
                    });

                    // 几何中心 P
                    DrawCross(canvas, globalP, new SKColor(255, 40, 40, 240), 3f, 14f);

                    using var textBg = new SKPaint
                    {
                        Color = new SKColor(0, 0, 0, 170),
                        Style = SKPaintStyle.Fill,
                        IsAntialias = true
                    };
                    using var textPaint = new SKPaint
                    {
                        Color = new SKColor(255, 100, 100, 255),
                        TextSize = Math.Max(18f, Math.Min(original.Width, original.Height) * 0.022f),
                        IsAntialias = true
                    };

                    string txt = $"P ({globalP.X:F1}, {globalP.Y:F1})";
                    float tw = textPaint.MeasureText(txt);
                    float tx = globalP.X + 16f;
                    float ty = globalP.Y - 16f;

                    canvas.DrawRoundRect(
                        new SKRoundRect(new SKRect(tx - 8f, ty - textPaint.TextSize, tx + tw + 8f, ty + 8f), 6f),
                        textBg);
                    canvas.DrawText(txt, tx, ty, textPaint);
                }

                // 可选：把所有药片中心也轻微标出来，方便核对
                foreach (var c in allPillCenters)
                {
                    DrawCross(canvas, c, new SKColor(80, 180, 255, 180), 2f, 8f);
                }
            }

            foreach (var area in areas)
            {
                var areaRect = model.ToOriginalRect(area);

                SegInstance matchedPill = null;
                if (availablePills.Count > 0)
                {
                    // 优先采用实例化时的「测地分区归属」：语义模型只输出两类掩膜，
                    // 圈实例与药片的对应关系在 ZoneSegModel 里已经确定，比最近邻更可靠。
                    if (area.DiskIndex >= 0)
                        matchedPill = availablePills.FirstOrDefault(p => p.DiskIndex == area.DiskIndex);

                    matchedPill ??= availablePills
                        .OrderBy(p =>
                        {
                            var pr = model.ToOriginalRect(p);
                            return Math.Pow(pr.MidX - areaRect.MidX, 2) +
                                   Math.Pow(pr.MidY - areaRect.MidY, 2);
                        })
                        .First();

                    availablePills.Remove(matchedPill);
                }

                var cropped = CropAndDraw(
                    original,
                    area,
                    matchedPill,
                    areaRect,
                    model,
                    classifier,
                    hasGlobalP,
                    globalP,
                    out SKBitmap rawCrop);

                if (cropped != null)
                {
                    _pairs.Add(new PillAntibacterialPair
                    {
                        Pill = matchedPill,
                        AntibacterialRegion = area,
                        CroppedImage = cropped,
                        RawPillCrop = rawCrop
                    });
                }

                done++;
                progress?.Report((double)done / total);
            }

            if (_pairs.Count > 0)
                _currentIndex = 0;
        }

        private SKBitmap CropAndDraw(
            SKBitmap original,
            SegInstance area,
            SegInstance pill,
            SKRect areaRect,
            ZoneSegModel model,
            PillClassifier classifier,
            bool hasGlobalP,
            SKPoint globalPOriginal,
            out SKBitmap outRawPillCrop)
        {
            outRawPillCrop = null;

            try
            {
                const int Padding = 20;
                int origW = original.Width;
                int origH = original.Height;

                var unionRect = areaRect;
                if (pill != null)
                {
                    var pr = model.ToOriginalRect(pill);
                    unionRect = SKRect.Create(
                        Math.Min(areaRect.Left, pr.Left),
                        Math.Min(areaRect.Top, pr.Top),
                        Math.Max(areaRect.Right, pr.Right) - Math.Min(areaRect.Left, pr.Left),
                        Math.Max(areaRect.Bottom, pr.Bottom) - Math.Min(areaRect.Top, pr.Top));
                }

                int cropX = Math.Max(0, (int)(unionRect.Left - Padding));
                int cropY = Math.Max(0, (int)(unionRect.Top - Padding));
                int cropW = Math.Min((int)(unionRect.Width + Padding * 2), origW - cropX);
                int cropH = Math.Min((int)(unionRect.Height + Padding * 2), origH - cropY);
                if (cropW <= 0 || cropH <= 0) return null;

                var cropped = new SKBitmap(cropW, cropH);
                using (var cv = new SKCanvas(cropped))
                {
                    cv.DrawBitmap(original,
                        new SKRect(cropX, cropY, cropX + cropW, cropY + cropH),
                        new SKRect(0, 0, cropW, cropH));
                }

                float ox = -cropX;
                float oy = -cropY;

                using var canvas = new SKCanvas(cropped);

                // ── 绘制 Area（绿色）──
                var areaPts = model.GetContourPointsInOriginal(area);
                SKPoint areaCenter = default;
                float areaRadius = 0f;

                if (areaPts.Count >= 3)
                {
                    var shifted = areaPts.Select(p => new SKPoint(p.X + ox, p.Y + oy)).ToList();
                    var path = PointsToPath(shifted);

                    canvas.DrawPath(path, new SKPaint
                    {
                        Color = new SKColor(29, 158, 117, 50),
                        Style = SKPaintStyle.Fill,
                        IsAntialias = true
                    });

                    canvas.DrawPath(path, new SKPaint
                    {
                        Color = new SKColor(29, 158, 117),
                        Style = SKPaintStyle.Stroke,
                        StrokeWidth = 2f,
                        IsAntialias = true
                    });

                    // 外接圆
                    ComputeMinEnclosingCircle(shifted, out areaCenter, out areaRadius);

                    canvas.DrawCircle(areaCenter, areaRadius, new SKPaint
                    {
                        Color = new SKColor(255, 200, 0),
                        Style = SKPaintStyle.Stroke,
                        StrokeWidth = 1.5f,
                        IsAntialias = true,
                        PathEffect = SKPathEffect.CreateDash(new[] { 6f, 4f }, 0)
                    });

                    if (shifted.Count >= 5)
                        FitAndDrawEllipse(canvas, shifted);
                }

                // ── 绘制 Pill（蓝色）──
                SKPoint pillCenter = default;
                float pillDiam = 0f;

                if (pill != null)
                {
                    var pillPts = model.GetContourPointsInOriginal(pill);
                    if (pillPts.Count >= 3)
                    {
                        var shifted = pillPts.Select(p => new SKPoint(p.X + ox, p.Y + oy)).ToList();
                        var path = PointsToPath(shifted);

                        canvas.DrawPath(path, new SKPaint
                        {
                            Color = new SKColor(24, 95, 165, 50),
                            Style = SKPaintStyle.Fill,
                            IsAntialias = true
                        });

                        canvas.DrawPath(path, new SKPaint
                        {
                            Color = new SKColor(24, 95, 165),
                            Style = SKPaintStyle.Stroke,
                            StrokeWidth = 2f,
                            IsAntialias = true
                        });

                        float bboxMinX = shifted.Min(p => p.X);
                        float bboxMaxX = shifted.Max(p => p.X);
                        float bboxMinY = shifted.Min(p => p.Y);
                        float bboxMaxY = shifted.Max(p => p.Y);
                        float bboxW = bboxMaxX - bboxMinX;
                        float bboxH = bboxMaxY - bboxMinY;

                        pillCenter = new SKPoint(
                            (bboxMinX + bboxMaxX) / 2f,
                            (bboxMinY + bboxMaxY) / 2f);

                        float pillRadius = Math.Max(bboxW, bboxH) / 2f;
                        pillDiam = pillRadius * 2f;

                        var bboxRect = new SKRect(bboxMinX, bboxMinY, bboxMaxX, bboxMaxY);
                        canvas.DrawRect(bboxRect, new SKPaint
                        {
                            Color = new SKColor(24, 95, 165),
                            Style = SKPaintStyle.Stroke,
                            StrokeWidth = 1f,
                            IsAntialias = true,
                            PathEffect = SKPathEffect.CreateDash(new[] { 4f, 3f }, 0)
                        });

                        DrawCross(canvas, pillCenter, new SKColor(24, 95, 165, 200), 1.5f, 8);
                    }
                    else
                    {
                        var pr = model.ToOriginalRect(pill);
                        pillCenter = new SKPoint(pr.MidX + ox, pr.MidY + oy);
                        pillDiam = (pr.Width + pr.Height) / 2f;
                    }
                }

                // ── 全局几何中心 P（转换到当前裁剪图坐标）──
                SKPoint globalPShifted = default;
                if (hasGlobalP)
                {
                    globalPShifted = new SKPoint(globalPOriginal.X + ox, globalPOriginal.Y + oy);
                    DrawCross(canvas, globalPShifted, new SKColor(255, 80, 80, 220), 2f, 10);
                }

                // ── 新规则：C -> P 方向 (±10°) 与 Area 实际边界(多边形)的平均交点距离 ──
                float finalInhibRatio = 0f;
                float finalInhibDist = 0f;
                float finalPillRadius = pillDiam / 2f;

                // 提取当前区域的实际多边形轮廓点
                var polyPts = areaPts.Select(p => new SKPoint(p.X + ox, p.Y + oy)).ToList();

                if (pill != null && polyPts.Count >= 3)
                {
                    // (可选) 绘制 areaCenter -> pillCenter 的轻微辅助线
                    if (areaRadius > 0)
                    {
                        canvas.DrawLine(areaCenter, pillCenter, new SKPaint
                        {
                            Color = new SKColor(255, 255, 255, 120),
                            StrokeWidth = 1f,
                            IsAntialias = true,
                            PathEffect = SKPathEffect.CreateDash(new[] { 5f, 3f }, 0)
                        });
                        DrawCross(canvas, areaCenter, new SKColor(255, 200, 0, 200), 1.5f, 8);
                    }

                    if (hasGlobalP)
                    {
                        // 1. 计算 C -> P 的基础主轴角度
                        float dx = globalPShifted.X - pillCenter.X;
                        float dy = globalPShifted.Y - pillCenter.Y;
                        float baseAngle = MathF.Atan2(dy, dx);

                        // 2. 在 ±10° 范围内采样 (21 条射线，每 1° 扫描一次)
                        float tenDegRad = 10f * MathF.PI / 180f;
                        int samples = 21;
                        float totalDist = 0f;
                        int validHits = 0;
                        var hitPoints = new List<SKPoint>();

                        for (int i = 0; i < samples; i++)
                        {
                            // 从 -10° 均匀过渡到 +10°
                            float angleOffset = -tenDegRad + (i * (2f * tenDegRad) / (samples - 1));
                            float currentAngle = baseAngle + angleOffset;

                            if (TryGetRayPolygonIntersection(pillCenter, currentAngle, polyPts, out SKPoint hitPt, out float hitDist))
                            {
                                totalDist += hitDist;
                                validHits++;
                                hitPoints.Add(hitPt); // 收集交点用于绘制可视化扇形
                            }
                        }

                        // 3. 计算平均值并绘制结果
                        if (validHits > 0)
                        {
                            finalInhibDist = totalDist / validHits;
                            finalInhibRatio = finalPillRadius > 0f
                                ? (finalInhibDist * 2f) / (finalPillRadius * 2f)
                                : 0f;

                            // 绘制 C -> P 基础主轴线 (红色虚线)
                            canvas.DrawLine(pillCenter, globalPShifted, new SKPaint
                            {
                                Color = new SKColor(255, 120, 120, 180),
                                StrokeWidth = 1.5f,
                                IsAntialias = true,
                                PathEffect = SKPathEffect.CreateDash(new[] { 7f, 4f }, 0)
                            });

                            // 绘制扇形采样区域 (半透明淡黄色，直观展示采样的实际轮廓)
                            if (hitPoints.Count > 1)
                            {
                                using var fanPaint = new SKPaint
                                {
                                    Color = new SKColor(255, 215, 60, 60), // 半透明黄色
                                    Style = SKPaintStyle.Fill,
                                    IsAntialias = true
                                };
                                var fanPath = new SKPath();
                                fanPath.MoveTo(pillCenter);
                                foreach (var hp in hitPoints) fanPath.LineTo(hp);
                                fanPath.Close();
                                canvas.DrawPath(fanPath, fanPaint);
                            }

                            // 计算平均半径对应的等效交点 Q (落在 C->P 主轴上)
                            SKPoint avgQ = new SKPoint(
                                pillCenter.X + MathF.Cos(baseAngle) * finalInhibDist,
                                pillCenter.Y + MathF.Sin(baseAngle) * finalInhibDist
                            );

                            // 绘制中心主轴的平均距离线段 (加粗黄线)
                            canvas.DrawLine(pillCenter, avgQ, new SKPaint
                            {
                                Color = new SKColor(255, 215, 60, 240),
                                StrokeWidth = 2.2f,
                                IsAntialias = true
                            });

                            // 在平均距离的尽头画一个明显的十字
                            DrawCross(canvas, avgQ, new SKColor(255, 215, 60, 230), 1.5f, 7);
                        }
                    }
                }

                // ── 测量标注 ──
                if (areaRadius > 0)
                {
                    var shiftedAreaPts = areaPts.Select(p => new SKPoint(p.X + ox, p.Y + oy)).ToList();
                    float areaMaskPx = Math.Abs(PolygonArea(shiftedAreaPts));
                    float eqDiam = 2f * MathF.Sqrt(areaMaskPx / MathF.PI);

                    DrawMeasurements(
                        canvas,
                        areaCenter,
                        areaRadius,
                        eqDiam,
                        areaRadius * 2f,
                        pillCenter,
                        finalPillRadius,
                        finalInhibDist,
                        finalInhibRatio,
                        cropW,
                        cropH);
                }

                // ══════════════════════════════════════════════
                // 左侧面板
                // ══════════════════════════════════════════════
                var combined = cropped;
                SKBitmap rawPillCrop = null;

                if (pill != null && finalPillRadius > 1f)
                {
                    const int PillPad = 16;
                    int pillCropSize = (int)(finalPillRadius * 2f + PillPad * 2);
                    int leftW = Math.Min(pillCropSize, cropH / 2);
                    int leftH = cropH;

                    float fs = leftW * 0.11f;
                    float fsVal = leftW * 0.14f;
                    float fsHi = leftW * 0.19f;

                    float rowGap = fsVal * 0.5f;
                    float innerGap = fsVal * 0.15f;
                    float textPadTop = fsVal * 0.5f;
                    float textTotalH = textPadTop
                        + (fs + innerGap + fsVal + rowGap) * 3
                        + (fsHi + innerGap + fsHi + rowGap);

                    int imgSize = Math.Max(Math.Min(leftW, (int)(leftH - textTotalH - 4)), leftW / 3);

                    string pillLabel = "?";
                    float pillConf = 0f;
                    SKBitmap displayBmp = null;

                    float pillCx = pillCenter.X + cropX;
                    float pillCy = pillCenter.Y + cropY;
                    int px0 = Math.Max(0, (int)(pillCx - finalPillRadius - PillPad));
                    int py0 = Math.Max(0, (int)(pillCy - finalPillRadius - PillPad));
                    int pw = Math.Min(pillCropSize, origW - px0);
                    int ph = Math.Min(pillCropSize, origH - py0);

                    if (pw > 0 && ph > 0)
                    {
                        var pillRaw = new SKBitmap(pw, ph);
                        using (var pc = new SKCanvas(pillRaw))
                        {
                            pc.DrawBitmap(original,
                                new SKRect(px0, py0, px0 + pw, py0 + ph),
                                new SKRect(0, 0, pw, ph));
                        }

                        rawPillCrop = pillRaw.Copy();

                        if (classifier != null)
                        {
                            (pillLabel, pillConf, displayBmp) = classifier.ClassifyWithPreview(pillRaw);
                            pillRaw.Dispose();
                        }
                        else
                        {
                            displayBmp = pillRaw;
                        }
                    }

                    var leftBmp = new SKBitmap(leftW, leftH);
                    using (var lc = new SKCanvas(leftBmp))
                    {
                        lc.Clear(new SKColor(22, 22, 22));

                        if (displayBmp != null)
                        {
                            lc.DrawBitmap(displayBmp,
                                new SKRect(0, 0, displayBmp.Width, displayBmp.Height),
                                new SKRect(0, 0, imgSize, imgSize));
                            displayBmp.Dispose();
                            displayBmp = null;
                        }

                        lc.DrawLine(0, imgSize + 4, leftW, imgSize + 4,
                            new SKPaint
                            {
                                Color = new SKColor(70, 70, 70),
                                StrokeWidth = Math.Max(1f, leftW * 0.005f)
                            });

                        var bold = SKTypeface.FromFamilyName("Arial",
                            SKFontStyleWeight.Bold,
                            SKFontStyleWidth.Normal,
                            SKFontStyleSlant.Upright);

                        var pLbl = new SKPaint
                        {
                            Color = new SKColor(190, 190, 190),
                            TextSize = fs,
                            IsAntialias = true,
                            Typeface = bold
                        };
                        var pVal = new SKPaint
                        {
                            Color = new SKColor(100, 225, 175),
                            TextSize = fsVal,
                            IsAntialias = true,
                            Typeface = bold
                        };
                        var pHiL = new SKPaint
                        {
                            Color = new SKColor(255, 230, 120),
                            TextSize = fsHi,
                            IsAntialias = true,
                            Typeface = bold
                        };
                        var pHiV = new SKPaint
                        {
                            Color = new SKColor(255, 210, 50),
                            TextSize = fsHi,
                            IsAntialias = true,
                            Typeface = bold
                        };
                        var pClsV = new SKPaint
                        {
                            Color = new SKColor(80, 220, 255),
                            TextSize = fsVal,
                            IsAntialias = true,
                            Typeface = bold
                        };

                        float unit = finalPillRadius > 0f
                            ? (finalInhibDist * 2f) / (finalPillRadius * 2f) * 6f
                            : 0f;

                        float tx = leftW * 0.05f;
                        float ty = imgSize + 4 + textPadTop;

                        lc.DrawText("Disk Dia", tx, ty + fs, pLbl);
                        ty += fs + innerGap;
                        lc.DrawText($"{finalPillRadius * 2f:F1} px", tx, ty + fsVal, pVal);
                        ty += fsVal + rowGap;

                        lc.DrawText("Ratio", tx, ty + fs, pLbl);
                        ty += fs + innerGap;
                        lc.DrawText($"{finalInhibRatio:F3}", tx, ty + fsVal, pVal);
                        ty += fsVal + rowGap;

                        lc.DrawText("Result", tx, ty + fsHi, pHiL);
                        ty += fsHi + innerGap;
                        lc.DrawText($"{unit:F2}", tx, ty + fsHi, pHiV);
                        ty += fsHi + rowGap;

                        lc.DrawText("Class", tx, ty + fs, pLbl);
                        ty += fs + innerGap;
                        lc.DrawText(classifier != null ? $"{pillLabel}  {pillConf * 100f:F0}%" : "N/A",
                            tx, ty + fsVal, pClsV);
                    }

                    combined = new SKBitmap(leftW + cropW, cropH);
                    using (var cc = new SKCanvas(combined))
                    {
                        cc.DrawBitmap(leftBmp, 0, 0);
                        cc.DrawBitmap(cropped, leftW, 0);
                        cc.DrawLine(leftW, 0, leftW, cropH,
                            new SKPaint { Color = new SKColor(90, 90, 90), StrokeWidth = 2f });
                    }

                    leftBmp.Dispose();
                    cropped.Dispose();
                }

                outRawPillCrop = rawPillCrop;
                return combined;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"CropAndDraw: {ex.Message}\n{ex.StackTrace}");
                return null;
            }
        }

        // ── 工具方法 ──────────────────────────────────────────
        private static SKPath PointsToPath(List<SKPoint> pts)
        {
            var path = new SKPath();
            path.MoveTo(pts[0]);
            foreach (var pt in pts.Skip(1))
                path.LineTo(pt);
            path.Close();
            return path;
        }

        private static void DrawCross(SKCanvas canvas, SKPoint center, SKColor color, float sw, float len)
        {
            using var p = new SKPaint { Color = color, StrokeWidth = sw, IsAntialias = true };
            canvas.DrawLine(center.X - len, center.Y, center.X + len, center.Y, p);
            canvas.DrawLine(center.X, center.Y - len, center.X, center.Y + len, p);
        }

        /// <summary>
        /// 从 start 出发，沿 towardPoint 方向的射线，
        /// 与圆(circleCenter, circleRadius)的前向交点
        /// </summary>
        private static bool TryGetRayCircleIntersection(
            SKPoint start,
            SKPoint towardPoint,
            SKPoint circleCenter,
            float circleRadius,
            out SKPoint hitPoint,
            out float hitDistance)
        {
            hitPoint = default;
            hitDistance = 0f;

            float vx = towardPoint.X - start.X;
            float vy = towardPoint.Y - start.Y;
            float vLen = MathF.Sqrt(vx * vx + vy * vy);

            if (vLen < 1e-6f)
            {
                vx = circleCenter.X - start.X;
                vy = circleCenter.Y - start.Y;
                vLen = MathF.Sqrt(vx * vx + vy * vy);
                if (vLen < 1e-6f)
                    return false;
            }

            float ux = vx / vLen;
            float uy = vy / vLen;

            float sx = start.X - circleCenter.X;
            float sy = start.Y - circleCenter.Y;

            float b = 2f * (sx * ux + sy * uy);
            float c = sx * sx + sy * sy - circleRadius * circleRadius;
            float disc = b * b - 4f * c;
            if (disc < 0f) return false;

            float sqrtDisc = MathF.Sqrt(disc);
            float t1 = (-b - sqrtDisc) / 2f;
            float t2 = (-b + sqrtDisc) / 2f;

            float t = Math.Max(t1, t2);
            if (t < 0f) return false;

            hitPoint = new SKPoint(start.X + ux * t, start.Y + uy * t);
            hitDistance = t;
            return true;
        }

        /// <summary>
        /// Ritter 近似最小外接圆算法
        /// </summary>
        private static void ComputeMinEnclosingCircle(List<SKPoint> pts, out SKPoint center, out float radius)
        {
            if (pts == null || pts.Count == 0) { center = default; radius = 0f; return; }
            if (pts.Count == 1) { center = pts[0]; radius = 0f; return; }

            SKPoint p1 = FarthestFrom(pts, pts[0]);
            SKPoint p2 = FarthestFrom(pts, p1);

            center = new SKPoint((p1.X + p2.X) / 2f, (p1.Y + p2.Y) / 2f);
            radius = Dist(p1, p2) / 2f;

            bool changed = true;
            while (changed)
            {
                changed = false;
                foreach (var p in pts)
                {
                    float d = Dist(p, center);
                    if (d > radius + 1e-4f)
                    {
                        float newRadius = (radius + d) / 2f;
                        float k = (d - newRadius) / d;
                        center = new SKPoint(
                            center.X + (p.X - center.X) * k,
                            center.Y + (p.Y - center.Y) * k);
                        radius = newRadius;
                        changed = true;
                    }
                }
            }
        }

        private static SKPoint FarthestFrom(List<SKPoint> pts, SKPoint origin)
        {
            SKPoint best = pts[0];
            float bestD = 0f;
            foreach (var p in pts)
            {
                float d = Dist(p, origin);
                if (d > bestD)
                {
                    bestD = d;
                    best = p;
                }
            }
            return best;
        }

        private static float Dist(SKPoint a, SKPoint b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return MathF.Sqrt(dx * dx + dy * dy);
        }

        private static void FitAndDrawEllipse(SKCanvas canvas, List<SKPoint> pts)
        {
            if (pts == null || pts.Count < 5) return;

            float cx = pts.Average(p => p.X);
            float cy = pts.Average(p => p.Y);

            float mxx = 0f, myy = 0f, mxy = 0f;
            foreach (var p in pts)
            {
                float dx = p.X - cx;
                float dy = p.Y - cy;
                mxx += dx * dx;
                myy += dy * dy;
                mxy += dx * dy;
            }

            mxx /= pts.Count;
            myy /= pts.Count;
            mxy /= pts.Count;

            float trace = mxx + myy;
            float det = mxx * myy - mxy * mxy;
            float disc = MathF.Sqrt(Math.Max(trace * trace / 4f - det, 0f));

            float a = 2f * MathF.Sqrt(Math.Max(trace / 2f + disc, 1f));
            float b = 2f * MathF.Sqrt(Math.Max(trace / 2f - disc, 1f));
            float angle = (Math.Abs(mxy) > 1e-6f || Math.Abs(mxx - myy) > 1e-6f)
                ? MathF.Atan2(2f * mxy, mxx - myy) / 2f
                : 0f;

            var path = new SKPath();
            for (int i = 0; i <= 72; i++)
            {
                float t = 2f * MathF.PI * i / 72;
                float ex = a * MathF.Cos(t);
                float ey = b * MathF.Sin(t);
                float rx = ex * MathF.Cos(angle) - ey * MathF.Sin(angle) + cx;
                float ry = ex * MathF.Sin(angle) + ey * MathF.Cos(angle) + cy;

                if (i == 0) path.MoveTo(rx, ry);
                else path.LineTo(rx, ry);
            }
            path.Close();

            canvas.DrawPath(path, new SKPaint
            {
                Color = new SKColor(255, 100, 200),
                Style = SKPaintStyle.Stroke,
                StrokeWidth = 1.5f,
                IsAntialias = true,
                PathEffect = SKPathEffect.CreateDash(new[] { 8f, 5f }, 0)
            });
        }

        private void DrawMeasurements(
            SKCanvas canvas,
            SKPoint areaCenter,
            float areaRadius,
            float eqDiam,
            float encDiam,
            SKPoint pillCenter,
            float pillRadius,
            float inhibDist,
            float inhibRatio,
            int imageW,
            int imageH)
        {
            bool hasPill = pillRadius > 0;
            float centerDist = 0f;

            if (hasPill)
            {
                float dx = areaCenter.X - pillCenter.X;
                float dy = areaCenter.Y - pillCenter.Y;
                centerDist = MathF.Sqrt(dx * dx + dy * dy);
            }

            var lines = new List<(string lbl, string val, bool hi)>
            {
                ("Equiv.D:",  $"{eqDiam:F1} px",  false),
                ("Enc.Circ:", $"{encDiam:F1} px", false),
            };

            if (hasPill)
            {
                lines.Add(("Pill Dia:", $"{pillRadius * 2f:F1} px", false));
                lines.Add(("Ctr Dist:", $"{centerDist:F1} px", false));
                lines.Add(("Inh.Dist:", $"{inhibDist:F1} px", true));
                lines.Add(("Dist/R:", $"{inhibRatio:F3}", true));
            }

            float fs = Math.Min(imageW, imageH) * 0.018f;
            float lineH = fs * 1.55f;
            float valOff = fs * 5.2f;
            float boxW = fs * 9.5f;
            float boxPad = fs * 0.6f;
            float boxH = lines.Count * lineH + boxPad * 2f;
            float startX = areaCenter.X + areaRadius + fs * 0.6f;
            float startY = areaCenter.Y - boxH / 2f;

            canvas.DrawRoundRect(new SKRoundRect(
                new SKRect(startX - boxPad, startY, startX + boxW, startY + boxH), fs * 0.4f),
                new SKPaint
                {
                    Color = new SKColor(0, 0, 0, 185),
                    Style = SKPaintStyle.Fill
                });

            var pLbl = new SKPaint
            {
                Color = new SKColor(190, 190, 190),
                TextSize = fs,
                IsAntialias = true
            };
            var pVal = new SKPaint
            {
                Color = new SKColor(100, 230, 180),
                TextSize = fs,
                IsAntialias = true
            };
            var pHi = new SKPaint
            {
                Color = new SKColor(255, 215, 60),
                TextSize = fs * 1.1f,
                IsAntialias = true
            };

            for (int i = 0; i < lines.Count; i++)
            {
                float y = startY + boxPad + fs + i * lineH;
                canvas.DrawText(lines[i].lbl, startX, y, lines[i].hi ? pHi : pLbl);
                canvas.DrawText(lines[i].val, startX + valOff, y, lines[i].hi ? pHi : pVal);
            }

            DrawCross(canvas, areaCenter, new SKColor(255, 200, 0, 220), 4f, 8);
        }

        private float PolygonArea(List<SKPoint> pts)
        {
            float area = 0;
            int n = pts.Count;
            for (int i = 0; i < n; i++)
            {
                var a = pts[i];
                var b = pts[(i + 1) % n];
                area += a.X * b.Y - b.X * a.Y;
            }
            return area / 2f;
        }

        // ── 导航 ──────────────────────────────────────────────
        public List<SKBitmap> GetAllCroppedRegions()
            => _pairs.Select(p => p.CroppedImage).Where(b => b != null).ToList();

        public List<SKBitmap> GetAllRawPillCrops()
            => _pairs.Select(p => p.RawPillCrop).Where(b => b != null).ToList();

        public SKBitmap GetCurrentCroppedRegion()
            => (_currentIndex >= 0 && _currentIndex < _pairs.Count)
               ? _pairs[_currentIndex].CroppedImage
               : null;

        public SKBitmap GetNextCroppedRegion()
        {
            if (_pairs.Count == 0) return null;
            _currentIndex = Math.Min(_currentIndex + 1, _pairs.Count - 1);
            return _pairs[_currentIndex].CroppedImage;
        }

        public SKBitmap GetPreviousCroppedRegion()
        {
            if (_pairs.Count == 0) return null;
            _currentIndex = Math.Max(_currentIndex - 1, 0);
            return _pairs[_currentIndex].CroppedImage;
        }

        public void Clear()
        {
            foreach (var pair in _pairs)
            {
                pair.CroppedImage?.Dispose();
                pair.RawPillCrop?.Dispose();
            }
            _pairs.Clear();
            _currentIndex = -1;

            _markedOriginalImage?.Dispose();
            _markedOriginalImage = null;
        }

        /// <summary>
        /// 射线与多边形求交点（获取射线方向上的最远交点，代表外边缘）
        /// </summary>
        private static bool TryGetRayPolygonIntersection(
            SKPoint start, float angle, List<SKPoint> polygon, out SKPoint hitPoint, out float hitDistance)
        {
            hitPoint = default;
            hitDistance = 0f;
            if (polygon == null || polygon.Count < 3) return false;

            // 射线的方向向量
            float dx = MathF.Cos(angle);
            float dy = MathF.Sin(angle);

            float maxDist = -1f;
            SKPoint bestPoint = default;
            int n = polygon.Count;

            // 遍历多边形的每一条边
            for (int i = 0; i < n; i++)
            {
                var p1 = polygon[i];
                var p2 = polygon[(i + 1) % n];

                float x1 = start.X, y1 = start.Y;
                float x2 = start.X + dx, y2 = start.Y + dy;
                float x3 = p1.X, y3 = p1.Y;
                float x4 = p2.X, y4 = p2.Y;

                // 计算线段求交行列式
                float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
                if (MathF.Abs(den) < 1e-6f) continue; // 平行无交点

                float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
                float u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;

                // t > 0 表示在射线前方； 0 <= u <= 1 表示交点真实落在线段 p1-p2 上
                if (t > 0 && u >= 0 && u <= 1)
                {
                    // 取射线方向的最远交点，这样可以穿透内部可能存在的空洞，直达最外侧轮廓
                    if (t > maxDist)
                    {
                        maxDist = t;
                        bestPoint = new SKPoint(x1 + t * dx, y1 + t * dy);
                    }
                }
            }

            if (maxDist > 0)
            {
                hitDistance = maxDist;
                hitPoint = bestPoint;
                return true;
            }

            return false;
        }

    }



    public class PillAntibacterialPair
    {
        public SegInstance Pill { get; set; }
        public SegInstance AntibacterialRegion { get; set; }
        public SKBitmap CroppedImage { get; set; }

        /// <summary>
        /// 未经任何处理的原始药片裁剪图（彩色，来自原图）
        /// </summary>
        public SKBitmap RawPillCrop { get; set; }
    }
}