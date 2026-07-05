from src.single_agent.single_agent_rag import answer

question = "Were there any soldiers who served at Verdun?" # chnage this question for testing.

result = answer(question)

print("Answer:", result["answer"])
