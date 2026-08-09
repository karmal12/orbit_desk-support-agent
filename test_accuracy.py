from graph import app

test_questions = [
    "What is OrbitDesk?",
    
]

for question in test_questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    result = app.invoke({
        "question": question
    })

    print("ANSWER:", result["answer"])