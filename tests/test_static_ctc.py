import numpy as np
import pytest
from static_ctc import decode_ctc,logical_persian


def test_ctc_blanks_and_duplicate_tokens_preserve_literal_digits():
    characters=['','0','1','2']
    indices=[1,1,0,1,2,2,3,0]
    predictions=np.full((1,len(indices),len(characters)),.01)
    for i,index in enumerate(indices): predictions[0,i,index]=.97
    text,score=decode_ctc(predictions,characters)
    assert text=='0012'
    assert score==pytest.approx(.97)


def test_blank_output_is_empty_with_zero_score():
    assert decode_ctc(np.array([[[.99,.01],[.99,.01]]]),['','1'])==('',0.0)


def test_wrong_dictionary_shape_is_rejected():
    with pytest.raises(ValueError): decode_ctc(np.zeros((1,3,4)),['','1'])


def test_persian_logical_order_preserves_date_and_leading_zero_runs():
    assert logical_persian('اضر')=='رضا'
    assert logical_persian('0012345678')=='0012345678'
    assert logical_persian('1370/01/02')=='1370/01/02'
