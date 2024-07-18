from tqdm.auto import tqdm
import time
import random
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter

##Define the tool
PAIR_SCHEMA = {
  'type': 'object',
  'properties': {
    'pair': {
      'type': 'object',
     'required': ["question", "answer"],
      'properties': {
        'question': {
          'type':'string',
          'description': 'A short question that can be answered from the chunk'
        },
       'answer': {
          'type':'string',
          'description':'a sentence solely from the document that answers accurately the question'
        }
      }
    }
  }
}


TOOL_DESC = """Generate  a pair of question and answer, based on a text chunk . It must include: 
- a question that can be answered from a document
- a passage from the text chunk that accurately answers the question, called 'answer'
"""

TOOL_SPEC = {"toolSpec":{
    "name":"pair_generator", 
    "description":TOOL_DESC, 
    "inputSchema":
        {"json":PAIR_SCHEMA}
    }
}

TOOL_LIST = [TOOL_SPEC]

###Define the generator
class SyntheticDataGenerator:
    def __init__(self, session, document_path, modelId , previous_questions = []):
        self.client = session.client('bedrock-runtime')
        self.text_chunks = self.read_10k_earnings(document_path)
        self.questions_per_chunk = {i:0 for  i in range(len(self.text_chunks))} #How many questions were asked per each chunk.
        self.privilege_questions_nb = -1 #We begin with documents which hadn't been selected yet, if not we increment.
        self.previous_questions = previous_questions
        self.modelId = modelId
        self.pairs = []

    def build_base_prompt(self, chunk):

        specified_prompt = f""".
        Based on the text chunk below, generate one question - answer pair that has a question, alongside a relevant passage from the text chunk.        <text_chunk>
        <text_chunk>
        {chunk}
        </text_chunk>

        Question needs to be specific.
        """

        return specified_prompt

    def atomic_invoke(self, chunk):
        base_prompt = self.build_base_prompt(chunk)
        msg ={"role":"user", "content":[{"text":base_prompt}]}

        system_prompt = """You are a helpful expert in financial information retrieval. 
        You excel at asking questions from 10k earnings that are diverse and know how to answer them with 10k earning documents.
        You know how to minimize bias.
        Your goal is to ask realistic and specific questions that can likely be asked for automated financial reporting.
        """
        
        system_prompt_conf = [{"text":system_prompt}]
        inf_config={"maxTokens": 4096,"temperature": 0.2}
        tool_conf = {"tools":TOOL_LIST, "toolChoice": {"tool": {"name":"pair_generator"}}}
        params = dict(modelId=self.modelId, messages = [msg], inferenceConfig = inf_config, toolConfig=tool_conf, system=system_prompt_conf)

        response = self.client.converse(**params)

        return response
    

    def validate_output(self, pair):
        return True ##Placeholder
    
    def post_process(self, response):
        response_message = response['output']['message']
        final_pair = response_message['content'][0]['toolUse']['input']['pair']
        if self.validate_output(final_pair):
            self.pairs.append(final_pair)
            self.previous_questions.append(final_pair['question'])
    
    def generate_one_pair(self, chunk):
        response = self.atomic_invoke(chunk)
        self.post_process(response)
    
    def generate_pairs(self, N = 50):
        attempts = 0
        current_eligible_indices = []
        for idx in tqdm(range(N)):
            if (len(self.pairs) >= N) or (attempts >=2*N):
                break

            try:
                attempts+=1                
                
                while len(current_eligible_indices)==0:
                    current_eligible_indices = [k for k,v in self.questions_per_chunk.items() if v<=self.privilege_questions_nb]
                    self.privilege_questions_nb+=1

                chunk_idx = random.choice(current_eligible_indices)
                chunk= self.text_chunks[chunk_idx]
                self.generate_one_pair(chunk)                    
            except Exception as err:
                print(err)
                time.sleep(30)
                continue
        
        ## We really want to ensure pair generation
        self.pairs = [v for v in self.pairs if isinstance(v, dict)]

    
    @staticmethod
    def read_10k_earnings(document_path):
        loader = PyPDFLoader(document_path, extract_images=False)
        splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1024, chunk_overlap=100)
        pages = loader.load_and_split(splitter)
        return [p.page_content for p in pages]
        

