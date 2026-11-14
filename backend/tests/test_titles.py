from app.schemas import Chunk
from app.search.titles import context_title, descriptive_title


def test_fallback_title_describes_evidence():
    source=Chunk(lecture_id='lec',start=51,end=60,text='Variables receive memory when they are created.')
    assert context_title([source])=='Variables receive memory when they are…'
    assert descriptive_title('Part 1',[source])==context_title([source])
    assert descriptive_title('Memory management',[source])=='Memory management'


def test_long_transcript_is_not_used_as_entire_label():
    source=Chunk(lecture_id='lec',start=0,end=30,text='Memory ' * 100)
    title=context_title([source])
    assert len(title.split())==6 and title.endswith('…')
