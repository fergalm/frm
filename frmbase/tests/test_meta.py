
import frmbase.meta as fmeta

def test_metadata_to_text():
    data = {
        'a': 1,
        'b': "two",
        'c': {
                'aa': 1,
                'bb': 2,
        }
    }


    text = fmeta.metadata_to_textlist(data)
    return text