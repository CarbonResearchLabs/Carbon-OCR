"""carbon_ocr, the C++ crop rule, detector targets and decoding for Python: the
same bytes as Pillow's bilinear resize, and the same decoding the engine runs."""
import numpy as np
import pytest
from PIL import Image

carbon_ocr = pytest.importorskip("carbon_ocr")


def test_rule_constants_and_region():
    assert (carbon_ocr.HEIGHT, carbon_ocr.WIDTH_STEP, carbon_ocr.FRAME) == (32, 8, 4)
    (x0, y0, x1, y1), w = carbon_ocr.crop_region((100, 50, 400, 80))
    assert (y0, y1, w) == (44.0, 86.0, 248)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(w / 32)
    assert carbon_ocr.ctc_frames((100, 50, 400, 80)) == 62
    assert carbon_ocr.crop_region((100, 50, 400, 80), height=48)[1] > w


def test_resample_is_pillows_bilinear():
    """Byte for byte, on rectangles inside and off the image, whenever Pillow runs its
    horizontal pass first (it does for line canvases; the C++ always does)."""
    rng = np.random.default_rng(0)
    h_img, w_img, py, px = 300, 900, 64, 640
    img = rng.integers(150, 256, (h_img, w_img), dtype=np.uint8)
    img[::7, :] = rng.integers(0, 80, (len(range(0, h_img, 7)), w_img), dtype=np.uint8)
    big = np.full((h_img + 2 * py, w_img + 2 * px), 255, np.uint8)
    big[py:-py, px:-px] = img
    pil_img = Image.fromarray(big)
    cases = 0
    while cases < 60:
        # coordinates in 1/64 px are exact in Pillow's float32 box, before and after the shift
        h = rng.integers(4 * 64, 120 * 64) / 64
        w = rng.integers(64, 700 * 64) / 64
        x0 = rng.integers(-20 * 64, int((w_img - w + 20) * 64)) / 64
        y0 = rng.integers(-20 * 64, int((h_img - h + 20) * 64)) / 64
        ow = max(8, int(round(w / h * 32 / 8)) * 8)
        if big.shape[0] - 32 > 2 * (big.shape[1] - ow):
            continue
        cases += 1
        ours = carbon_ocr.resample(img, (x0, y0, x0 + w, y0 + h), ow, 32)
        pil = np.asarray(pil_img.resize((ow, 32), Image.Resampling.BILINEAR,
                                        box=(x0 + px, y0 + py, x0 + w + px, y0 + h + py)))
        assert np.array_equal(ours, pil), (x0, y0, w, h, ow)


def test_crop_and_crop_lines_agree():
    img = np.full((200, 600), 255, np.uint8)
    img[70:100, 150:450] = 0
    one = carbon_ocr.crop(img, (150, 70, 450, 100))
    assert one.shape == (32, 248) and one.dtype == np.uint8 and one[16, 124] == 0 and one[1, 1] == 255
    batch, widths = carbon_ocr.crop_lines(img, np.array([[150, 70, 450, 100], [10, 5, 60, 30]], float), 400)
    assert batch.shape == (2, 32, 400) and widths == [248, carbon_ocr.crop_region((10, 5, 60, 30))[1]]
    assert np.array_equal(batch[0, :, :248], one)
    assert (batch[0, :, 248:] == carbon_ocr.ground(one)).all()
    with pytest.raises(RuntimeError):
        carbon_ocr.crop_lines(img, np.array([[150, 70, 450, 100.0]]), 200)


def test_det_targets_and_decode_round_trip():
    lines = np.array([[40.0, 30, 380, 52], [40, 62, 300, 84], [420, 30, 760, 47]])
    pictures = np.array([[420.0, 100, 760, 400]])
    maps = carbon_ocr.det_targets(lines, pictures, 512, 400)
    assert maps.shape == (4, 512, 400) and maps.dtype == np.float32
    assert set(np.unique(maps[0])) == {0.0, 1.0}
    logits = maps.copy()
    for k in (0, 3):
        logits[k] = np.where(maps[k] > 0.5, 8.0, -8.0)
    got, pics = carbon_ocr.decode_det(logits, scale=2.0)
    assert len(got) == 3 and len(pics) == 1
    want = sorted(lines.tolist(), key=lambda b: (b[1], b[0]))
    for (box, score, pixels), w in zip(got, want):
        assert abs(box[1] - 2 * w[1]) < 1e-3 and abs(box[3] - 2 * w[3]) < 1e-3
        assert abs(box[0] - 2 * w[0]) <= 2.0 and abs(box[2] - 2 * w[2]) <= 2.0
        assert score > 0.99 and pixels > 0
    assert abs(pics[0][0][0] - 840) <= 2 and abs(pics[0][0][3] - 800) <= 2


def test_det_decoded_on_the_gpu_is_the_cpus():
    """cuda_decode_det thresholds a batch of maps where they are, a torch tensor's memory, and
    decodes the masks: the boxes decode_det gives from the whole maps, page by page."""
    torch = pytest.importorskip("torch")
    if not (carbon_ocr.HAS_CUDA and torch.cuda.is_available()):
        pytest.skip("needs carbon_ocr built with CUDA and a CUDA device")
    rng = np.random.default_rng(3)
    pages = []
    for _ in range(3):
        top = np.sort(rng.uniform(20, 900, 12))
        lines = np.stack([rng.uniform(20, 200, 12), top, rng.uniform(300, 760, 12), top + rng.uniform(12, 30, 12)], 1)
        maps = carbon_ocr.det_targets(lines, np.array([[420.0, 920, 760, 1000]]), 512, 400)
        for k in (0, 3):
            maps[k] = np.where(maps[k] > 0.5, 6.0, -6.0) + rng.normal(0, 3, maps[k].shape)
        pages.append(maps.astype(np.float32))
    batch = torch.from_numpy(np.stack(pages)).cuda()
    got = carbon_ocr.cuda_decode_det(batch.data_ptr(), 3, 512, 400, stream=torch.cuda.current_stream().cuda_stream)
    for maps, (lines, pictures) in zip(pages, got):
        assert (lines, pictures) == carbon_ocr.decode_det(maps)
        assert lines


def test_text_errors_are_pythons_edit_distances():
    """CER and WER counts in C++ (carbon/score.h) against a Levenshtein in Python over code points and
    str.split() words, on lines a few edits from their truth: ASCII, accents, scripts, odd white space."""
    def distance(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    rng = np.random.default_rng(5)
    alphabet = list("abc deéüकि中\U0001f600") + [" ", "　", "​", "\t"]
    truths, preds = [], []
    for _ in range(300):
        t = "".join(rng.choice(alphabet, rng.integers(0, 40)))
        p = list(t)
        for _ in range(rng.integers(0, 5)):
            at = int(rng.integers(0, len(p) + 1))
            p[at:at + int(rng.integers(0, 2))] = list(rng.choice(alphabet, rng.integers(0, 2)))
        truths.append(t)
        preds.append("".join(p))
    got = carbon_ocr.text_errors(truths, preds)
    assert got.shape == (300, 4) and got.dtype == np.int64
    want = [(distance(t, p), len(t), distance(t.split(), p.split()), len(t.split())) for t, p in zip(truths, preds)]
    assert got.tolist() == [list(w) for w in want]
    ids = [[int(c) for c in rng.integers(0, 800, rng.integers(0, 30))] for _ in range(50)]
    other = [s[:3] + [7] + s[5:] for s in ids]
    assert carbon_ocr.edit_distances(ids, other).tolist() == [distance(a, b) for a, b in zip(ids, other)]
    with pytest.raises(RuntimeError):
        carbon_ocr.text_errors(["a"], [])


def test_ctc_greedy_and_collapse():
    logits = np.zeros((1, 4, 6), np.float32)
    for t, c in enumerate([0, 2, 2, 0, 2, 3]):
        logits[0, c, t] = 5.0
    (ids, frames, probs), = carbon_ocr.ctc_greedy(logits, np.array([6], np.int32))
    assert ids == [2, 2, 3] and frames == [1, 4, 5]
    p = np.exp(5.0) / (np.exp(5.0) + 3)
    assert probs == pytest.approx([p] * 3, rel=1e-5)
    best = logits.argmax(1).astype(np.int32)
    (ids2, frames2, _), = carbon_ocr.ctc_collapse(best, np.ones_like(best, np.float32), np.array([6], np.int32))
    assert (ids2, frames2) == (ids, frames)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(5, 30, 40)).astype(np.float32)
    lengths = rng.integers(0, 41, 5).astype(np.int32)
    full = carbon_ocr.ctc_greedy(z, lengths)
    part = carbon_ocr.ctc_collapse(z.argmax(1).astype(np.int32), np.ones((5, 40), np.float32), lengths)
    assert [l[0] for l in full] == [l[0] for l in part]


DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


def test_tensor_round_trips_every_dtype():
    for dt in (np.float32, np.float16, np.int32, np.int64, np.uint8, np.bool_):
        a = (np.arange(24) % 7).astype(dt).reshape(2, 3, 4)
        t = carbon_ocr.Tensor(a)
        assert t.shape == [2, 3, 4]
        b = t.numpy()
        assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b)
    assert carbon_ocr.Tensor(np.zeros((1, 2), np.float32)).dtype == "float32"
    with pytest.raises((TypeError, RuntimeError)):
        carbon_ocr.Tensor(np.zeros(3, np.complex64))


def test_session_runs_a_model_from_numpy():
    s = carbon_ocr.Session(str(DATA / "colsum_argmax.onnx"))
    assert s.backend == "ort"
    assert s.inputs == [("x", "float32", [-1, 1, -1, -1])]
    assert [o[0] for o in s.outputs] == ["colsum", "ids"]
    x = np.random.default_rng(0).normal(size=(3, 1, 5, 7)).astype(np.float32)
    colsum, ids = s.run([x])
    want = np.maximum(x, 0).sum(axis=2)
    assert np.allclose(colsum, want, atol=1e-5) and (ids == want.argmax(axis=2)).all()
    by_name = s.run({"x": carbon_ocr.Tensor(x)})  # a dict by name, a Tensor in it
    assert np.array_equal(by_name[0], colsum)
    with pytest.raises(RuntimeError):
        s.run([np.zeros((3, 2, 5, 7), np.float32)])  # the channel dimension is fixed at 1
    with pytest.raises(RuntimeError):
        carbon_ocr.Session(str(DATA / "no-such.onnx"))


def test_engine_reads_a_page():
    page = carbon_ocr.load_image(str(DATA / "fake-set" / "page.png"))
    assert page.shape == (1024, 800) and page.dtype == np.uint8
    e = carbon_ocr.Engine(str(DATA / "fake-set"), workers=2)
    assert e.workers == 2
    out = e.ocr([page, page])
    assert [l["text"] for l in out[0]["lines"]] == ["aaa", "aaaaa", "aa"]
    assert out[1]["text"] == "aaa\naaaaa\naa\n"
    assert abs(out[0]["lines"][0]["box"][1] - 200) < 2 and out[0]["ms"]["total"] > 0
    assert out[0]["order"] == "xy-cut" and out[0]["blocks"] == []  # the fake set has no order model
    # tests/data/fake-order.onnx: not sure of its order, so the XY-cut's, but its blocks
    ordered = carbon_ocr.Engine(str(DATA / "fake-set"), workers=1, order_model=str(DATA / "fake-order.onnx"))
    o = ordered.ocr([page])[0]
    assert o["order"] == "xy-cut" and [l["text"] for l in o["lines"]] == ["aaa", "aaaaa", "aa"]
    assert [(b["label"], b["lines"]) for b in o["blocks"]] == [("Section-Header", [i]) for i in range(3)]


def test_the_batched_pipeline_reads_as_the_engine_does():
    # carbon::Pipeline on the CPU: each model run on a batch from the pages in flight, page for page the engine's
    page = carbon_ocr.load_image(str(DATA / "fake-set" / "page.png"))
    blank = np.full((300, 400), 255, np.uint8)
    pages = [page, blank, page, page, blank]
    p = carbon_ocr.Pipeline(str(DATA / "fake-set"), device="cpu", page_batch=4, threads=2, model_threads=2)
    assert p.device == "cpu"
    out = p.ocr(pages)
    want = carbon_ocr.Engine(str(DATA / "fake-set"), workers=2).ocr(pages)
    read = "aaa\naaaaa\naa\n"
    assert [o["text"] for o in out] == [w["text"] for w in want] == [read, "", read, read, ""]
    assert [[l["box"] for l in o["lines"]] for o in out] == [[l["box"] for l in w["lines"]] for w in want]
    s = p.stats
    assert s["pages"] == 5 and s["lines"] == 9 and 1 <= s["detect_runs"] <= 5
    with pytest.raises(Exception):
        carbon_ocr.Pipeline(str(DATA / "fake-set"), device="tpu")
