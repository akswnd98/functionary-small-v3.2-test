from typing import List, Tuple, Any
from langchain_core.agents import AgentAction, AgentFinish
from langchain_community.llms.llamacpp import LlamaCpp
from transformers import PreTrainedTokenizer
from json import dumps, loads
from langchain.agents.agent import AgentOutputParser
from tools import get_current_weather, send_email, search_web

def generate_history_passthrough_function ():
  def history_passthrough_function (inputs):
    # format
    # {'name': 'get_current_weather', 'arguments': {'location': 'Seoul, KR', 'format': 'celsius'}}
    intermediate_steps: List[Tuple[AgentAction, str]] = inputs['intermediate_steps']
    input: str = inputs['input']
    ret = [
      {'role': 'user', 'content': input}
    ]
    for intermediate_step in intermediate_steps:
      tool_call = {'name': intermediate_step[0].tool, 'arguments': dumps(intermediate_step[0].tool_input)}
      ret += [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'function': tool_call}]},
        {'role': 'tool', 'name': intermediate_step[0].tool, 'content': intermediate_step[1]}
      ]

    return ret

  return history_passthrough_function

def generate_llm_function (tokenizer: PreTrainedTokenizer, llm: LlamaCpp):
  tools = [get_current_weather.func, send_email.func, search_web.func]

  def llm_function (inputs):
    history: List[dict[str, Any]] = inputs['history']
    llm_inputs = tokenizer.apply_chat_template(
      conversation=history,
      tools=tools,
      add_generation_prompt=True,
      return_dict=True,
      return_tensors="pt",
    )
    ret = llm.invoke(tokenizer.decode(llm_inputs['input_ids'][0]))
    try:
      ret = loads(ret)
    except Exception as e:
      return ret

    return dumps(ret[0])

  return llm_function

class FunctionaryAgentOutputParser (AgentOutputParser):
  def parse (self, text: str) -> AgentAction | AgentFinish:
    try:
      rst = text.split('>>>')
      rst = list(map(lambda x: x.split('\n'), rst))
      rst = [[x[0], loads(x[1])] for x in rst]
    except Exception as e:
      print(e)
      return AgentFinish({'output': text}, text)

    # if jsonified['name'] == 'finish':
      # return AgentFinish({'output': jsonified['arguments']['final_answer']}, jsonified['arguments']['final_answer'])

    return AgentAction(rst[0][0], rst[0][1], 'no log')
