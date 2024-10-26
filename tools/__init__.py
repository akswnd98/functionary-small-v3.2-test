from langchain_core.tools import tool

@tool
def get_current_weather (location: str, format: str):
  """
  Get the current weather

  Args:
    location: The city and state, e.g. San Francisco, CA
    format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
  """
  return '38 degree'

@tool
def send_email (address: str, subject: str, body: str):
  """
  send email

  Args:
    address: target email address.
    subject: title of this email.
    body: main detail of this email.
  """
  return 'email sent successfully'

@tool
def search_web (text: str):
  """
  search web about text

  Args:
    text: subject to search about.
  """
  return 'the average day time temperature of seoul is 32 degree celsius'
