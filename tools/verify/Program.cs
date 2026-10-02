// 轻量核验程序：在不需要 MAUI 工作负载的情况下，编译并运行 App 里的
// ZoneSegModel / PillClassifier，把它们的结果与 Python 参考实现对比。
//
//   dotnet run --project "Antibacterial zone/_verify/Verify.csproj" -- seg <onnx> <图片或目录> <输出目录>
//   dotnet run --project "Antibacterial zone/_verify/Verify.csproj" -- cls <onnx> <裁剪图目录> <输出json>
//
using Antibacterial_zone;
using SkiaSharp;
using System.Text.Json;

string Cmd = args.Length > 0 ? args[0] : "";
if (args.Length == 0)
{
    Console.WriteLine("usage: Verify seg|cls ...");
    return 2;
}

switch (Cmd)
{
    case "seg":
    {
        string onnx = args[1], input = args[2], outDir = args[3];
        bool noTta = args.Contains("--no-tta");
        ZoneSegModel.UseTta = !noTta;
        Directory.CreateDirectory(outDir);
        Directory.CreateDirectory(Path.Combine(outDir, "masks"));

        var model = new ZoneSegModel(File.ReadAllBytes(onnx));
        var images = new List<string>();
        if (Directory.Exists(input))
            images.AddRange(Directory.GetFiles(input, "*.*")
                .Where(f => new[] { ".jpg", ".jpeg", ".png", ".bmp" }
                    .Contains(Path.GetExtension(f).ToLowerInvariant()))
                .OrderBy(f => f));
        else
            images.Add(input);

        var rows = new List<object>();
        foreach (var img in images)
        {
            string stem = Path.GetFileNameWithoutExtension(img);
            using var bmp = SKBitmap.Decode(img);
            if (bmp == null) { Console.Error.WriteLine($"decode failed: {img}"); continue; }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var pre = model.PreprocessImage(bmp);
            var preds = model.Predict(pre);
            sw.Stop();

            var list = new List<object>();
            foreach (var p in preds)
            {
                var rect = model.ToOriginalRect(p);
                list.Add(new
                {
                    cls = p.ClassName,
                    cls_id = p.ClassId,
                    conf = Math.Round(p.Confidence, 4),
                    disk_index = p.DiskIndex,
                    has_zone = p.HasZone,
                    mask_area_px = p.Mask.Count(v => v != 0),
                    eq_diameter_px = Math.Round(model.EquivalentDiameter(p), 2),
                    bbox_orig = new[]
                    {
                        Math.Round(rect.Left, 1), Math.Round(rect.Top, 1),
                        Math.Round(rect.Right, 1), Math.Round(rect.Bottom, 1)
                    },
                    contour_pts = model.GetContourPointsInOriginal(p).Count,
                });

                // 导出 640 二值掩膜，供 Python 端算 IoU
                int w = p.MaskW, h = p.MaskH;
                using var mb = new SKBitmap(w, h);
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        byte v = p.Mask[y * w + x];
                        mb.SetPixel(x, y, v != 0 ? SKColors.White : SKColors.Black);
                    }
                string mp = Path.Combine(outDir, "masks",
                    $"{stem}__{p.ClassName}_{p.DiskIndex}.png");
                using (var fs2 = File.Create(mp)) mb.Encode(fs2, SKEncodedImageFormat.Png, 100);
            }

            // 可视化
            using (var vis = model.DrawPredictions(bmp, preds))
            using (var fs3 = File.Create(Path.Combine(outDir, $"{stem}__vis.jpg")))
                vis.Encode(fs3, SKEncodedImageFormat.Jpeg, 90);

            rows.Add(new
            {
                image = stem,
                width = bmp.Width,
                height = bmp.Height,
                ms = sw.ElapsedMilliseconds,
                n_pred = preds.Count,
                n_disk = preds.Count(p => p.ClassId == 1),
                n_zone = preds.Count(p => p.ClassId == 0),
                orphan_zone_px = model.LastOrphanZonePixels,
                instances = list,
            });
            Console.WriteLine($"{stem}: {preds.Count(p => p.ClassId == 1)} disks, " +
                              $"{preds.Count(p => p.ClassId == 0)} zones, {sw.ElapsedMilliseconds} ms");
        }

        File.WriteAllText(Path.Combine(outDir, "seg_csharp.json"),
            JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"wrote {Path.Combine(outDir, "seg_csharp.json")}");
        return 0;
    }

    case "cls":
    {
        string onnx = args[1], dir = args[2], outJson = args[3];
        var clf = new PillClassifier(File.ReadAllBytes(onnx));
        var files = Directory.GetFiles(dir, "*.*")
            .Where(f => new[] { ".png", ".jpg", ".jpeg", ".bmp" }
                .Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f).ToList();

        string enhDir = Path.Combine(Path.GetDirectoryName(outJson) ?? ".", "enhanced_csharp");
        Directory.CreateDirectory(enhDir);

        var rows = new List<object>();
        foreach (var f in files)
        {
            using var bmp = SKBitmap.Decode(f);
            if (bmp == null) continue;
            var (label, conf, disp) = clf.ClassifyWithPreview(bmp);
            if (disp != null)
            {
                using var fs = File.Create(Path.Combine(enhDir,
                    Path.GetFileNameWithoutExtension(f) + ".png"));
                disp.Encode(fs, SKEncodedImageFormat.Png, 100);
                disp.Dispose();
            }
            rows.Add(new
            {
                file = Path.GetFileName(f),
                label,
                confidence = Math.Round(conf, 4),
                w = bmp.Width,
                h = bmp.Height,
            });
            Console.WriteLine($"{Path.GetFileName(f)} -> {label} {conf:P1}");
        }

        File.WriteAllText(outJson,
            JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"wrote {outJson}  ({rows.Count} files)");
        return 0;
    }

    default:
        Console.WriteLine("unknown command");
        return 2;
}
