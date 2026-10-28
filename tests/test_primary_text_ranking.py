from local_worker import select_engine_consensus


def test_strong_stable_primary_text_wins_review_tie_without_becoming_accepted():
    text,ambiguous,votes,evidence,source=select_engine_consensus(
        {'bina_rizeh_ppocrv6':['تنسسرین']*3,'paddleocr_v5_arabic':['نسرین']*3},
        {'bina_rizeh_ppocrv6':[.62,.68,.70],'paddleocr_v5_arabic':[.95,.96,.97]},'first_name')
    assert text=='نسرین' and source=='paddleocr_v5_arabic'
    assert ambiguous and len(votes)==2
    assert evidence['bina_rizeh_ppocrv6']['value']=='تنسسرین'


def test_two_engines_still_outrank_one_even_when_the_single_engine_is_strong():
    text,ambiguous,_,_,_=select_engine_consensus(
        {'bina_rizeh_ppocrv6':['محمد']*3,'paddleocr_v5_arabic':['احمد']*3,'paddle_names':['محمد']*3},
        {'bina_rizeh_ppocrv6':[.8]*3,'paddleocr_v5_arabic':[.99]*3,'paddle_names':[.85]*3},'first_name')
    assert text=='محمد' and not ambiguous
