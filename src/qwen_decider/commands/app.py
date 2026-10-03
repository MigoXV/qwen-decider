import logging
from pathlib import Path

import typer

from qwen_decider.settings import Settings

app = typer.Typer(no_args_is_help=True)


@app.command()
def serve(
    model_path: Path = typer.Option(..., envvar="DECIDER_MODEL_PATH"),
    precision: str = typer.Option("bf16", envvar="DECIDER_PRECISION"),
    host: str = typer.Option("127.0.0.1", envvar="DECIDER_HOST"),
    port: int = typer.Option(10003, envvar="DECIDER_PORT"),
    max_num_seqs: int = typer.Option(1, envvar="DECIDER_MAX_NUM_SEQS"),
    max_requests: int = typer.Option(16, envvar="DECIDER_MAX_REQUESTS"),
    request_timeout: float = typer.Option(30, envvar="DECIDER_REQUEST_TIMEOUT"),
    gpu_memory_utilization: float = typer.Option(0.12, envvar="DECIDER_GPU_MEMORY_UTILIZATION"),
    enforce_eager: bool = typer.Option(False, envvar="DECIDER_ENFORCE_EAGER"),
):
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    import uvicorn

    from qwen_decider.api import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings(
        model_path=model_path,
        precision=precision,
        max_num_seqs=max_num_seqs,
        max_requests=max_requests,
        request_timeout=request_timeout,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    config = uvicorn.Config(
        create_app(settings), host=host, port=port, workers=1, timeout_graceful_shutdown=5
    )
    with config.bind_socket() as sock:
        sock.listen(config.backlog)
        uvicorn.Server(config).run(sockets=[sock])


@app.command("convert")
def convert_command(
    source: Path = typer.Argument(...),
    destination: Path = typer.Argument(...),
    precision: str = typer.Option("bf16"),
):
    if precision not in {"bf16", "fp16", "fp32"}:
        raise typer.BadParameter("precision must be bf16, fp16 or fp32")
    from qwen_decider.convert import convert

    convert(source, destination, precision)
    typer.echo(str(destination))


@app.command("quantize-int8")
def quantize_int8(source: Path = typer.Argument(...), destination: Path = typer.Argument(...)):
    from qwen_decider.quantize import quantize

    quantize(source, destination)
    typer.echo(str(destination))


if __name__ == "__main__":
    app()
