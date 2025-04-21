from openai import OpenAI
client = OpenAI()

# response = client.files.create(
#   file=open("fine-tuning-data.jsonl", "rb"),
#   purpose="fine-tune"
# )


# client.fine_tuning.jobs.create(
#   training_file=response.id,
#   model="gpt-4o-mini-2024-07-18"
# )

job_status = client.fine_tuning.jobs.retrieve('INSERT-JOB-ID')
print(job_status)