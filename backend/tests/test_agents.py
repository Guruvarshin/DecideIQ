"""
Phase 5 test: LangGraph question generation agent on two job offer documents.
Run: docker exec decideiq-backend-1 python tests/test_agents_phase5.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
from app.agents.question_generator import generate_questions

FILES = [
    ("/app/data/job_offers/offer_a_techcorp.txt", "TechCorp Offer"),
    ("/app/data/job_offers/offer_b_finedge.txt", "FinEdge Offer"),
]


async def main():
    for path, name in FILES:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        print(f"\n{'='*60}")
        print(f"Document: {name}")
        print(f"Words   : {len(text.split())}")
        print(f"{'='*60}")

        questions = await generate_questions(text, name)

        print(f"\nGenerated {len(questions)} questions:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")

    print("\nDone.")


asyncio.run(main())
