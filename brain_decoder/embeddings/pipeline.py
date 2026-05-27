from pathlib import Path

import numpy as np

from brain_decoder.embeddings.clip_hf import CLIPImageExtractor


def save_image_embeddings(
    image_dir: Path | str,
    output_path: Path | str,
    extractor: CLIPImageExtractor,
    *,
    pattern: str = "*.png",
) -> np.ndarray:
    paths = sorted(Path(image_dir).glob(pattern))
    embeddings = extractor.extract_batch(paths)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    return embeddings
