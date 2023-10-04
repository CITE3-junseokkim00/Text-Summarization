import torch
from transformers import T5Tokenizer
from transformers.models.t5 import T5ForConditionalGeneration

def load_model():
    model = T5ForConditionalGeneration.from_pretrained('./koT5_summary')
    return model
def load_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained('./vocab/sentencepiece.model')
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

text = """또는 아원자 입자 및 입자 집단을 다루는 현대 물리학의 기초 이론이다. '아무리 기이하고 터무니없는 사건이라 해도, 발생 확률이 0이 아닌 이상 반드시 일어난다'는 물리학적 아이디어에 기초한다.[1] 양자역학의 양자는 물리량에 기본 단위가 있으며, 그 기본 단위에 정수배만 존재한다는 뜻을 담고 있다. 현대 물리학의 기초인 양자역학은 컴퓨터의 주요 부품인 반도체의 원리를 설명해 주고, "물질의 운동이 본질적으로 비결정론적인가?" 라는 의문을 제기하며 과학기술, 철학, 문학, 예술 등 다방면에 중요한 영향을 미쳐 20세기 과학사에서 빼놓을 수 없는 중요한 이론으로 평가된다.
19세기 중반까지의 실험은 뉴턴의 고전역학으로 설명할 수 있었다. 그러나, 19세기 후반부터 20세기 초반까지 이루어진 전자, 양성자, 중성자 등의 아원자 입자와 관련된 실험들의 결과는 고전역학으로 설명을 시도할 경우 모순이 발생하여 이를 해결하기 위한 새로운 역학 체계가 필요하게 되었다. 이 양자역학은 플랑크의 양자 가설을 계기로 하여 슈뢰딩거, 하이젠베르크, 디랙 등에 의해 만들어진 전적으로 20세기에 이루어진 학문이다. 양자역학에서 플랑크 상수를 0으로 극한을 취하면 양자역학이 고전역학으로 수렴하는데, 이를 대응 원리라 한다.
"""

text = text.replace('\n','')
input_ids = tokenizer.encode(text)
input_ids = torch.tensor(input_ids)
input_ids = input_ids.unsqueeze(0)
output = model.generate(inputs=input_ids, eos_token_id=1, max_length=512, num_beams=5)
output = tokenizer.decode(output[0], skip_special_tokens=True)
print(output)