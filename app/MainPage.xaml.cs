using Microsoft.Maui.Storage;
using Microsoft.Maui.ApplicationModel;
using SkiaSharp;
using System.IO;

namespace Antibacterial_zone
{
    public partial class MainPage : ContentPage
    {
        private ZoneSegModel _model;
        private SKBitmap _originalBitmap;
        private string _selectedImagePath;
        private bool _modelReady = false;
        private bool _isAnalyzing = false;
        private readonly ImageProcessor _imageProcessor = new();
        private PillClassifier _classifier;
        // _resultImageTempPath 仅供 Export 功能读取文件，不再用于 Image 控件显示
        private string _resultImageTempPath;
        private string _inputImageTempPath;

        private string _sessionId = DateTime.Now.ToString("HHmmss");

        public MainPage()
        {
            InitializeComponent();
            CleanupAllCacheFiles();
            _ = InitializeModelAsync();
        }

        // ══════════════════════════════════════════
        //  ✅ 核心辅助方法：SKBitmap → ImageSource
        //  先用 SKImage.FromBitmap 固化像素快照，再编码为 bytes，
        //  最后用 FromStream 返回。平台渲染器每次拿到全新 MemoryStream，
        //  无法命中缓存，彻底避免显示旧图问题。
        // ══════════════════════════════════════════
        private static ImageSource GetImageSourceFromSKBitmap(SKBitmap bitmap)
        {
            using var image = SKImage.FromBitmap(bitmap);
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            var bytes = data.ToArray();
            return ImageSource.FromStream(() => new MemoryStream(bytes));
        }

        // ══════════════════════════════════════════
        //  模型初始化
        // ══════════════════════════════════════════
        private async Task InitializeModelAsync()
        {
            try
            {
                SetStatus("Loading model...");
                using var stream = await FileSystem.OpenAppPackageFileAsync("best.onnx");
                using var ms = new MemoryStream();
                await stream.CopyToAsync(ms);
                var bytes = ms.ToArray();
                await Task.Run(() => _model = new ZoneSegModel(bytes));

                using var clsStream = await FileSystem.OpenAppPackageFileAsync("class.onnx");
                using var clsMs = new MemoryStream();
                await clsStream.CopyToAsync(clsMs);
                var clsBytes = clsMs.ToArray();
                await Task.Run(() => _classifier = new PillClassifier(clsBytes));

                _modelReady = true;
                SetStatus("Ready");
            }
            catch (Exception ex)
            {
                SetStatus($"Model load failed: {ex.Message}");
            }
        }

        // ══════════════════════════════════════════
        //  打开图片
        // ══════════════════════════════════════════
        private async void OnOpenClicked(object? sender, EventArgs e)
        {
            try
            {
                var result = await FilePicker.PickAsync(new PickOptions
                {
                    PickerTitle = "Select image",
                    FileTypes = FilePickerFileType.Images
                });
                if (result == null) return;

                CleanupTempFiles();

                _selectedImagePath = result.FullPath;
                _originalBitmap?.Dispose();
                _originalBitmap = null;

                using var fs = File.OpenRead(_selectedImagePath);
                _originalBitmap = SKBitmap.Decode(fs);

                if (_originalBitmap == null) { SetStatus("Image decode failed"); return; }

                string uploadCache = Path.Combine(
                    FileSystem.CacheDirectory,
                    $"input_{DateTime.Now:yyyyMMdd_HHmmss}.jpg");
                File.Copy(_selectedImagePath, uploadCache, overwrite: true);
                _inputImageTempPath = uploadCache;
                _selectedImagePath = uploadCache;

                // ✅ 直接赋值，不做 null 重置
                SelectedImage.Source = GetImageSourceFromSKBitmap(_originalBitmap);

                CroppedImage.Source = null;
                ExportButton.IsEnabled = false;
                SaveRawButton.IsEnabled = false;
                _imageProcessor.Clear();
                UpdateNavigationButtons();
                SetStatus($"{Path.GetFileName(result.FullPath)}  " +
                          $"({_originalBitmap.Width} x {_originalBitmap.Height})");
            }
            catch (Exception ex)
            {
                SetStatus($"Open failed: {ex.Message}");
            }
        }

        // ══════════════════════════════════════════
        //  拍摄图片
        // ══════════════════════════════════════════
        private async void OnCaptureClicked(object? sender, EventArgs e)
        {
            try
            {
                var status = await Permissions.RequestAsync<Permissions.Camera>();
                if (status != PermissionStatus.Granted)
                {
                    SetStatus("Camera permission denied");
                    return;
                }

                System.Diagnostics.Debug.WriteLine(
                    $"[Capture] IsCaptureSupported={MediaPicker.Default.IsCaptureSupported}");
                if (!MediaPicker.Default.IsCaptureSupported)
                {
                    SetStatus("Camera not available (emulator?)");
                    return;
                }

                var photo = await MediaPicker.Default.CapturePhotoAsync(new MediaPickerOptions
                {
                    Title = "Capture sample"
                });
                if (photo == null) return;

                CleanupTempFiles();
                SetStatus("Saving photo...");

                string cachePath = Path.Combine(
                    FileSystem.CacheDirectory,
                    $"capture_{DateTime.Now:yyyyMMdd_HHmmss}.jpg");

                using (var srcStream = await photo.OpenReadAsync())
                using (var dstStream = File.Create(cachePath))
                {
                    await srcStream.CopyToAsync(dstStream);
                }

                _inputImageTempPath = cachePath;
                _selectedImagePath = cachePath;
                _originalBitmap?.Dispose();
                _originalBitmap = null;
                _originalBitmap = SKBitmap.Decode(cachePath);

                if (_originalBitmap == null) { SetStatus("Capture decode failed"); return; }

                // ✅ 直接赋值，不做 null 重置
                SelectedImage.Source = GetImageSourceFromSKBitmap(_originalBitmap);

                CroppedImage.Source = null;
                ExportButton.IsEnabled = false;
                SaveRawButton.IsEnabled = false;
                _imageProcessor.Clear();
                UpdateNavigationButtons();
                SetStatus($"Captured  ({_originalBitmap.Width} x {_originalBitmap.Height})");
            }
            catch (FeatureNotSupportedException fex)
            {
                SetStatus($"Not supported: {fex.Message}");
                System.Diagnostics.Debug.WriteLine($"[Capture] FeatureNotSupported: {fex}");
            }
            catch (PermissionException pex)
            {
                SetStatus($"Permission denied: {pex.Message}");
                System.Diagnostics.Debug.WriteLine($"[Capture] Permission: {pex}");
            }
            catch (Exception ex)
            {
                SetStatus($"Capture failed: {ex.GetType().Name}: {ex.Message}");
                System.Diagnostics.Debug.WriteLine($"[Capture] Exception: {ex}");
            }
        }

        // ══════════════════════════════════════════
        //  分析图片
        // ══════════════════════════════════════════
        private async void OnAnalyzeClicked(object? sender, EventArgs e)
        {
            if (_originalBitmap == null) { SetStatus("Please open an image first"); return; }
            if (!_modelReady || _model == null) { SetStatus("Model not ready"); return; }
            if (_isAnalyzing) return;

            _isAnalyzing = true;
            AnalyzeButton.IsEnabled = false;
            OpenButton.IsEnabled = false;
            CaptureButton.IsEnabled = false;

            AnalysisProgress.Progress = 0;
            AnalysisProgress.IsVisible = true;
            _sessionId = DateTime.Now.ToString("HHmmss");
            SetStatus("Preprocessing...");

            try
            {
                // ── 步骤1：预处理 ──
                SKBitmap preprocessed;
                try
                {
                    preprocessed = _model.PreprocessImage(_originalBitmap);
                }
                catch (Exception ex)
                {
                    SetStatus($"Preprocess failed: {ex.Message}");
                    return;
                }

                await AnalysisProgress.ProgressTo(0.1, 200, Easing.Linear);
                SetStatus("Running inference.");

                // ── 步骤2：ONNX 推理 ──
                var dotsCts = new CancellationTokenSource();
                var dotsToken = dotsCts.Token;
                int dotsCount = 1;
                var dotsTimer = new System.Threading.Timer(_ =>
                {
                    if (dotsToken.IsCancellationRequested) return;
                    dotsCount = dotsCount % 3 + 1;
                    SetStatus($"Running inference{new string('.', dotsCount)}");
                }, null, 400, 400);

                List<SegInstance> predictions;
                bool inferOk = false;
                try
                {
                    predictions = await Task.Run(() =>
                    {
                        var r = _model.Predict(preprocessed);
                        preprocessed.Dispose();
                        return r;
                    });
                    inferOk = true;
                }
                catch (Exception ex)
                {
                    preprocessed.Dispose();
                    SetStatus($"Inference failed: {ex.Message}");
                    predictions = null;
                }
                finally
                {
                    dotsCts.Cancel();
                    dotsTimer.Dispose();
                    dotsCts.Dispose();
                }

                if (!inferOk || predictions == null) return;

                await AnalysisProgress.ProgressTo(0.35, 200, Easing.Linear);

                if (predictions.Count == 0)
                {
                    SetStatus("No targets detected");
                    _imageProcessor.Clear();
                    UpdateNavigationButtons();
                    return;
                }

                // ── 步骤3：绘制主图 ──
                SetStatus("Drawing results...");
                SKBitmap resultBmp;
                try
                {
                    resultBmp = _model.DrawPredictions(_originalBitmap, predictions);
                }
                catch (Exception ex)
                {
                    SetStatus($"Draw failed: {ex.Message}");
                    return;
                }

                // 写文件仅供 Export 使用
                _resultImageTempPath = Path.Combine(
                    FileSystem.CacheDirectory,
                    $"result_{DateTime.Now:yyyyMMdd_HHmmss}.png");
                await Task.Run(() =>
                {
                    using var ms = new MemoryStream();
                    resultBmp.Encode(ms, SKEncodedImageFormat.Png, 100);
                    File.WriteAllBytes(_resultImageTempPath, ms.ToArray());
                });

                // ✅ 直接赋值，不做 null 重置，SKImage.FromBitmap 固化快照
                SelectedImage.Source = GetImageSourceFromSKBitmap(resultBmp);
                resultBmp.Dispose();

                await AnalysisProgress.ProgressTo(0.5, 150, Easing.Linear);
                SetStatus("Cropping regions...");

                // ── 步骤4：裁剪子图 ──
                int areasTotal = predictions.Count(p => p.ClassId == 0);
                var origRef = _originalBitmap;

                var progress = new Progress<double>(p =>
                {
                    AnalysisProgress.Progress = 0.5 + p * 0.5;
                    int done = (int)Math.Round(p * areasTotal);
                    SetStatus($"Cropping  {done} / {areasTotal}...");
                });

                try
                {
                    await Task.Run(() =>
                        _imageProcessor.ProcessDetectedRegions(
                            origRef, predictions, _model, progress, _classifier));
                }
                catch (Exception ex)
                {
                    SetStatus($"Crop failed: {ex.Message}");
                    return;
                }

                // ── 步骤5：显示结果 ──
                await AnalysisProgress.ProgressTo(1.0, 150, Easing.Linear);
                ShowCurrentCroppedRegion();
                UpdateNavigationButtons();

                int ac = predictions.Count(p => p.ClassId == 0);
                int tc = predictions.Count(p => p.ClassId == 1);
                SetStatus($"Done  —  {ac} areas, {tc} pills, {_imageProcessor.TotalRegions} pairs");

                bool hasResults = _imageProcessor.TotalRegions > 0;
                ExportButton.IsEnabled = hasResults;
                SaveRawButton.IsEnabled = hasResults;

                await Task.Delay(800);
                AnalysisProgress.IsVisible = false;
            }
            catch (Exception ex)
            {
                SetStatus($"Error: {ex.GetType().Name}: {ex.Message}");
            }
            finally
            {
                _isAnalyzing = false;
                AnalyzeButton.IsEnabled = true;
                OpenButton.IsEnabled = true;
                CaptureButton.IsEnabled = true;
            }
        }

        // ══════════════════════════════════════════
        //  翻页
        // ══════════════════════════════════════════
        private void OnPreviousClicked(object? sender, EventArgs e)
        {
            _imageProcessor.GetPreviousCroppedRegion();
            UpdateNavigationButtons();
            ShowCurrentCroppedRegion();
        }

        private void OnNextClicked(object? sender, EventArgs e)
        {
            _imageProcessor.GetNextCroppedRegion();
            UpdateNavigationButtons();
            ShowCurrentCroppedRegion();
        }

        // ══════════════════════════════════════════
        //  导出（分析结果长图）
        // ══════════════════════════════════════════
        private async void OnExportClicked(object? sender, EventArgs e)
        {
            if (_resultImageTempPath == null || !File.Exists(_resultImageTempPath))
            {
                SetStatus("No result to export, run Analyze first");
                return;
            }

            ExportButton.IsEnabled = false;
            SetStatus("Exporting...");

            try
            {
                string exportPath = await Task.Run(() =>
                {
                    var bitmaps = new List<SKBitmap>();

                    var resultBmp = SKBitmap.Decode(_resultImageTempPath);
                    if (resultBmp != null) bitmaps.Add(resultBmp);

                    bitmaps.AddRange(_imageProcessor.GetAllCroppedRegions()
                        .Select(b => b?.Copy())
                        .Where(b => b != null));

                    if (bitmaps.Count == 0) return null;

                    int maxW = bitmaps.Max(b => b.Width);
                    int totalH = 0;
                    var scaled = new List<SKBitmap>();

                    foreach (var bmp in bitmaps)
                    {
                        if (bmp.Width == maxW)
                        {
                            scaled.Add(bmp);
                            totalH += bmp.Height;
                        }
                        else
                        {
                            float ratio = (float)maxW / bmp.Width;
                            int newH = (int)(bmp.Height * ratio);
                            var resized = new SKBitmap(maxW, newH);
                            using var cv = new SKCanvas(resized);
                            cv.DrawBitmap(bmp,
                                new SKRect(0, 0, bmp.Width, bmp.Height),
                                new SKRect(0, 0, maxW, newH));
                            scaled.Add(resized);
                            totalH += newH;
                            bmp.Dispose();
                        }
                    }

                    var combined = new SKBitmap(maxW, totalH);
                    using (var canvas = new SKCanvas(combined))
                    {
                        canvas.Clear(new SKColor(20, 20, 20));
                        int y = 0;
                        foreach (var s in scaled)
                        {
                            canvas.DrawBitmap(s, 0, y);
                            y += s.Height;
                            s.Dispose();
                        }
                    }

                    string outPath = Path.Combine(
                        FileSystem.CacheDirectory,
                        $"export_{DateTime.Now:yyyyMMdd_HHmmss}.jpg");
                    using var ms = new MemoryStream();
                    combined.Encode(ms, SKEncodedImageFormat.Jpeg, 92);
                    combined.Dispose();
                    File.WriteAllBytes(outPath, ms.ToArray());
                    return outPath;
                });

                if (exportPath == null)
                {
                    SetStatus("Export failed: no images");
                    return;
                }

                await Share.RequestAsync(new ShareFileRequest
                {
                    Title = "Export analysis result",
                    File = new ShareFile(exportPath, "image/jpeg")
                });
                SetStatus("Export ready — save via share sheet");
            }
            catch (Exception ex)
            {
                SetStatus($"Export failed: {ex.Message}");
            }
            finally
            {
                ExportButton.IsEnabled = true;
            }
        }

        // ══════════════════════════════════════════
        //  Save Raw Crops
        // ══════════════════════════════════════════
        private async void OnSaveRawClicked(object? sender, EventArgs e)
        {
            var rawCrops = _imageProcessor.GetAllRawPillCrops();
            if (rawCrops == null || rawCrops.Count == 0)
            {
                SetStatus("No raw crops available, run Analyze first");
                return;
            }

            SaveRawButton.IsEnabled = false;
            SetStatus("Saving raw crops...");

            try
            {
                string zipPath = await Task.Run(() =>
                {
                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    string zipFile = Path.Combine(
                        FileSystem.CacheDirectory,
                        $"raw_crops_{timestamp}.zip");

                    using var zipStream = File.Create(zipFile);
                    using var archive = new System.IO.Compression.ZipArchive(
                        zipStream, System.IO.Compression.ZipArchiveMode.Create, leaveOpen: false);

                    for (int i = 0; i < rawCrops.Count; i++)
                    {
                        var bmp = rawCrops[i];
                        if (bmp == null || bmp.IsNull) continue;

                        string entryName = $"pill_{i + 1:D3}.png";
                        var entry = archive.CreateEntry(entryName,
                            System.IO.Compression.CompressionLevel.Fastest);

                        using var entryStream = entry.Open();
                        using var ms = new MemoryStream();
                        bmp.Encode(ms, SKEncodedImageFormat.Png, 100);
                        ms.Position = 0;
                        ms.CopyTo(entryStream);
                    }

                    return zipFile;
                });

                await Share.RequestAsync(new ShareFileRequest
                {
                    Title = $"Raw pill crops ({rawCrops.Count} images)",
                    File = new ShareFile(zipPath, "application/zip")
                });
                SetStatus($"Raw crops ready — {rawCrops.Count} images in ZIP");
            }
            catch (Exception ex)
            {
                SetStatus($"Save raw failed: {ex.Message}");
                System.Diagnostics.Debug.WriteLine($"[SaveRaw] {ex}");
            }
            finally
            {
                SaveRawButton.IsEnabled = true;
            }
        }

        // ══════════════════════════════════════════
        //  辅助方法
        // ══════════════════════════════════════════

        // ✅ 同步版本：直接调用 GetImageSourceFromSKBitmap 赋值，
        //    彻底去掉 async/await、Source=null、Task.Delay 这整个模式。
        private void ShowCurrentCroppedRegion()
        {
            var bmp = _imageProcessor.GetCurrentCroppedRegion();
            if (bmp == null || bmp.IsNull)
            {
                CroppedImage.Source = null;
                return;
            }
            CroppedImage.Source = GetImageSourceFromSKBitmap(bmp);
        }

        private void UpdateNavigationButtons()
        {
            bool has = _imageProcessor.TotalRegions > 0;
            PreviousButton.IsEnabled = has && _imageProcessor.CurrentIndex > 0;
            NextButton.IsEnabled = has && _imageProcessor.CurrentIndex < _imageProcessor.TotalRegions - 1;
            CropPositionLabel.Text = has
                ? $"{_imageProcessor.CurrentIndex + 1} / {_imageProcessor.TotalRegions}"
                : "0 / 0";
        }

        private void CleanupAllCacheFiles()
        {
            try
            {
                var dir = FileSystem.CacheDirectory;
                foreach (var prefix in new[] { "capture_", "input_", "result_", "cropped_", "captured_input", "raw_crops_", "export_" })
                {
                    foreach (var f in Directory.GetFiles(dir, $"{prefix}*"))
                    {
                        try { File.Delete(f); } catch { }
                    }
                }
            }
            catch { }
        }

        private void CleanupTempFiles()
        {
            foreach (var f in new[] { _inputImageTempPath, _resultImageTempPath })
            {
                if (f != null && File.Exists(f))
                    try { File.Delete(f); } catch { }
            }
            _inputImageTempPath = null;
            _resultImageTempPath = null;
        }

        private void SetStatus(string text)
        {
            if (MainThread.IsMainThread)
                StatusLabel.Text = text;
            else
                MainThread.BeginInvokeOnMainThread(() => StatusLabel.Text = text);
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            _classifier?.Dispose();
            _classifier = null;
            _model?.Dispose();
            _model = null;
        }
    }
}