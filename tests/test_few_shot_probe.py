"""Unit tests for the few-shot CLIP probe — no video files, no network, no
real CLIP model loaded (a fake embedder stands in for ClipEmbedder so these
stay fast and offline, consistent with the rest of tests/)."""
import tempfile
from pathlib import Path

import torch

from src.classifiers.few_shot_probe import (
    IMAGE_EXTS,
    MIN_EXAMPLES_PER_CLASS,
    VIDEO_EXTS,
    ClipEmbedder,
    LinearProbe,
    discover_few_shot_classes,
    load_probe,
    save_probe,
    score_with_probe,
    train_probe,
)


class FakeEmbedder:
    """Returns a fixed, separable embedding per "clip" (path) instead of
    running real CLIP — positive-labeled paths cluster near [1,0,...],
    negative-labeled paths near [-1,0,...], so a trained probe should reach
    high train accuracy, proving the training loop itself works correctly."""

    def __init__(self, embed_dim: int = 8):
        self.embed_dim = embed_dim

    def embed_video(self, path, n_frames=6):
        return self._embed(path, n_frames)

    def embed_image(self, path):
        return self._embed(path, n_frames=1)

    def embed_media(self, path, n_frames=6):
        return self._embed(path, n_frames)

    def _embed(self, path, n_frames):
        torch.manual_seed(hash(str(path)) % (2**31))
        is_positive = "positive" in str(path)
        center = torch.zeros(self.embed_dim)
        center[0] = 1.0 if is_positive else -1.0
        noise = torch.randn(n_frames, self.embed_dim) * 0.05
        embeds = center.unsqueeze(0) + noise
        return embeds / embeds.norm(dim=-1, keepdim=True)


def make_fake_clips(tmpdir: Path, n_positive: int, n_negative: int) -> tuple[list[Path], list[Path]]:
    pos = [tmpdir / f"positive_{i}.mp4" for i in range(n_positive)]
    neg = [tmpdir / f"negative_{i}.mp4" for i in range(n_negative)]
    for p in pos + neg:
        p.touch()
    return pos, neg


def test_linear_probe_forward_shape():
    probe = LinearProbe(embed_dim=16)
    x = torch.randn(5, 16)
    out = probe(x)
    assert out.shape == (5,)


def test_discover_few_shot_classes_finds_complete_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fire_smoke" / "positive").mkdir(parents=True)
        (tmp / "fire_smoke" / "negative").mkdir(parents=True)
        (tmp / "fire_smoke" / "positive" / "a.mp4").touch()
        (tmp / "fire_smoke" / "negative" / "b.mp4").touch()
        # incomplete: only positive/, no negative/ — must be skipped
        (tmp / "incomplete" / "positive").mkdir(parents=True)

        found = discover_few_shot_classes(tmp)
        assert set(found.keys()) == {"fire_smoke"}
        assert len(found["fire_smoke"]["positive"]) == 1
        assert len(found["fire_smoke"]["negative"]) == 1


def test_discover_few_shot_classes_empty_dir_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert discover_few_shot_classes(Path(tmp) / "does_not_exist") == {}


def test_train_probe_refuses_too_few_examples():
    embedder = FakeEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        pos, neg = make_fake_clips(Path(tmp), n_positive=2, n_negative=2)
        assert 2 < MIN_EXAMPLES_PER_CLASS  # sanity: this test is actually exercising the refusal
        result = train_probe(embedder, "fire_smoke", pos, neg, epochs=20)
        assert result is None


def test_train_probe_learns_separable_data():
    embedder = FakeEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        pos, neg = make_fake_clips(Path(tmp), n_positive=MIN_EXAMPLES_PER_CLASS, n_negative=MIN_EXAMPLES_PER_CLASS)
        result = train_probe(embedder, "fire_smoke", pos, neg, epochs=200)
        assert result is not None
        probe, stats = result
        assert stats.class_key == "fire_smoke"
        assert stats.train_accuracy > 0.9  # trivially separable synthetic data
        assert stats.num_positive_frames == MIN_EXAMPLES_PER_CLASS * 6  # FRAMES_PER_CLIP default


def test_probe_save_load_roundtrip_preserves_predictions():
    embedder = FakeEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pos, neg = make_fake_clips(tmp, n_positive=MIN_EXAMPLES_PER_CLASS, n_negative=MIN_EXAMPLES_PER_CLASS)
        probe, _ = train_probe(embedder, "fighting", pos, neg, epochs=100)

        test_embed = embedder.embed_video(tmp / "positive_0.mp4")[0]
        score_before = score_with_probe(probe, test_embed)

        checkpoint_dir = tmp / "checkpoints"
        saved_path = save_probe(probe, "fighting", checkpoint_dir)
        assert saved_path.exists()

        loaded = load_probe("fighting", checkpoint_dir)
        assert loaded is not None
        score_after = score_with_probe(loaded, test_embed)
        assert abs(score_before - score_after) < 1e-6


def test_load_probe_returns_none_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_probe("nonexistent_class", tmp) is None


def test_trained_probe_scores_positive_higher_than_negative():
    embedder = FakeEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pos, neg = make_fake_clips(tmp, n_positive=MIN_EXAMPLES_PER_CLASS, n_negative=MIN_EXAMPLES_PER_CLASS)
        probe, _ = train_probe(embedder, "vandalism", pos, neg, epochs=200)

        pos_embed = embedder.embed_video(tmp / "positive_0.mp4")[0]
        neg_embed = embedder.embed_video(tmp / "negative_0.mp4")[0]
        assert score_with_probe(probe, pos_embed) > score_with_probe(probe, neg_embed)


def test_discover_few_shot_classes_finds_images_and_videos_mixed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fighting" / "positive").mkdir(parents=True)
        (tmp / "fighting" / "negative").mkdir(parents=True)
        (tmp / "fighting" / "positive" / "clip.mp4").touch()
        (tmp / "fighting" / "positive" / "photo.jpg").touch()
        (tmp / "fighting" / "positive" / "photo2.png").touch()
        (tmp / "fighting" / "positive" / "not_media.txt").touch()  # must be ignored
        (tmp / "fighting" / "negative" / "other.webm").touch()

        found = discover_few_shot_classes(tmp)
        positive_names = {p.name for p in found["fighting"]["positive"]}
        assert positive_names == {"clip.mp4", "photo.jpg", "photo2.png"}


def test_image_and_video_extension_sets_are_disjoint():
    assert IMAGE_EXTS.isdisjoint(VIDEO_EXTS)


def test_embed_media_dispatches_to_image_for_image_extension():
    embedder = object.__new__(ClipEmbedder)  # skip __init__, no real CLIP load needed
    calls = []
    embedder.embed_image = lambda p: calls.append(("image", p)) or torch.zeros(1, 8)
    embedder.embed_video = lambda p, n_frames=6: calls.append(("video", p)) or torch.zeros(1, 8)

    embedder.embed_media(Path("photo.jpg"))
    embedder.embed_media(Path("clip.mp4"))

    assert calls == [("image", Path("photo.jpg")), ("video", Path("clip.mp4"))]


def test_embed_media_passes_n_frames_through_for_video():
    embedder = object.__new__(ClipEmbedder)
    calls = []
    embedder.embed_image = lambda p: torch.zeros(1, 8)
    embedder.embed_video = lambda p, n_frames=6: calls.append(n_frames) or torch.zeros(1, 8)

    embedder.embed_media(Path("clip.mp4"), n_frames=3)
    assert calls == [3]
