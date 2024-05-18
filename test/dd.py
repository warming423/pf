from typing import Optional

def test_optional(name:Optional[str]=None):
    if name is None:
        print("none")
    elif type(name) is str:
        print("str")
 
        
test_optional(1111)