import pytest
from app.ingestion.timing import source_id
from app.schemas import Chunk, AskResponse, AnswerSentence
from app.search import presentation, sections


@pytest.fixture
def example(monkeypatch):
    sources = [Chunk(lecture_id='lec', start=s, end=s+2, text=f'Evidence {s}.') for s in [51.56,54.46,57.48,111.3]]
    index = sections.build_index('lec', sources, [
        sections.SectionDraft(title='Memory', sentence_ids=[source_id(s) for s in sources[:3]]),
        sections.SectionDraft(title='Allocation', sentence_ids=[source_id(sources[3])])])
    monkeypatch.setattr(presentation.segmentation, 'ready', lambda *a: index)
    return sources, index


def test_nonconsecutive_references_deduplicate_globally_without_losing_evidence(example):
    sources, _ = example
    sentences = [AnswerSentence(text='First.',sources=[sources[1]]),
                 AnswerSentence(text='Another topic.',sources=[sources[3]]),
                 AnswerSentence(text='Back to memory.',sources=[sources[0],sources[2]])]
    answer = AskResponse(answer='Old', sources=sources, sentences=sentences)
    before = answer.model_dump()
    result = presentation.with_citations('lec',answer)
    assert [c.start for c in result.citations] == [51.56,111.3]
    kept = lambda dumped: [{k: s[k] for k in ('text', 'sources')} for s in dumped['sentences']]
    assert kept(result.model_dump()) == kept(before)
    first, second = (s.id for s in example[1].sections)
    assert [s.section_ids for s in result.sentences] == [[first], [second], [first]]
    assert result.model_dump()['sources'] == before['sources']
    assert result.answer.count('[0:51]') == 1
    assert '[0:54]' not in result.answer


def test_replay_starts_at_first_cited_sentence_not_section_start(example):
    sources, _ = example
    answer = AskResponse(answer='Claim',sources=[sources[2]],sentences=[AnswerSentence(text='Claim.',sources=[sources[2]])])
    assert presentation.with_citations('lec',answer).citations[0].start == 57.48


def test_adjacent_different_sections_stay_distinct(monkeypatch):
    sources = [Chunk(lecture_id='lec',start=s,end=s+1,text=str(s)) for s in [51,53]]
    index=sections.build_index('lec',sources,[sections.SectionDraft(title=str(i),sentence_ids=[source_id(s)]) for i,s in enumerate(sources)])
    monkeypatch.setattr(presentation.segmentation,'ready',lambda *a:index)
    answer=AskResponse(answer='Claim',sources=sources,sentences=[AnswerSentence(text='Claim.',sources=sources)])
    assert len(presentation.with_citations('lec',answer).citations)==2


def test_fallback_preserves_sentence_evidence(example, monkeypatch):
    sources,_=example
    monkeypatch.setattr(presentation,'section_index',lambda *a:None)
    answer=AskResponse(answer='Claim',sources=sources,sentences=[AnswerSentence(text='Claim.',sources=sources)])
    citations=presentation.with_citations('lec',answer).citations
    assert len(citations)==2
    assert answer.sentences[0].section_ids==[c.section_id for c in citations]
    assert answer.sources==sources


def test_timeline_and_answer_use_same_index(client,example):
    _,index=example
    response=client.get('/timeline',params={'lecture_id':'lec'})
    assert response.status_code==200
    assert [s['start'] for s in response.json()]==[s.start for s in index.sections]
    assert [s['title'] for s in response.json()]==['Memory','Allocation']


def test_ask_endpoint_preserves_all_evidence_and_emits_section_links(client,example,fake_llm,monkeypatch):
    from app.search import retrieval
    sources,index=example
    monkeypatch.setattr(retrieval,'transcript',lambda lid:sources)
    monkeypatch.setattr(retrieval,'search',lambda *a:sources)
    monkeypatch.setattr(retrieval,'answer_sources',lambda lid,hits:hits)
    fake_llm.reply_answer('groq',('Memory discussion.',sources[:3]),('Allocation example.',sources[3:]))
    body=client.post('/ask',json={'lecture_id':'lec','question':'where is memory mentioned?'}).json()
    assert body['sources']==[s.model_dump() for s in sources]
    assert len(body['citations'])==2
    assert body['citations'][0]['section_id']==index.sections[0].id
    assert body['citations'][0]['start']==51.56
    assert len(body['sentences'][0]['sources'])==3
