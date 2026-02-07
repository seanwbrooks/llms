import requests

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"

filename = url.split('/')[-1]

try:
	resp = requests.get(url)
	resp.raise_for_status()

	with open(filename, 'w') as file:
		file.write(resp.text)
except Exception as e:
	print("Error: ", e)
