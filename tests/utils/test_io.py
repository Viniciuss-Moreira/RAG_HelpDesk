from src.utils.io import read_txt_chunks
import tempfile

def test_read_txt_chunks():
    text = "Chunk 1\n\nChunk 2\n\nChunk 3"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(text)
        tmp.seek(0)
        result = read_txt_chunks(tmp.name)
    assert result == ["Chunk 1", "Chunk 2", "Chunk 3"]