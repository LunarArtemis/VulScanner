from joern import JoernClient
client = JoernClient()
slices = client.run_query('your_query.scala')