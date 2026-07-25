"""Run the deterministic SOP timeline without a browser."""
import asyncio
from app.main import app

async def main() -> None:
    service=app.state.service; service.start()
    for _ in range(60): await service.process_frame()
    print(service.status()); print(service.repository.events(service.session["id"]))
if __name__ == "__main__": asyncio.run(main())
