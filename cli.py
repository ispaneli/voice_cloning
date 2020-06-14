import click
from main import clone_voice as main_clone_voice

DEFAULT_MESSAGE = "Hi! You have launched a project to clone a voice. " \
                  "This is a test message. " \
                  "Thanks for your attention. " \
                  "See you later!"


@click.group()
def cli():
    pass


@click.command()
@click.argument('path_to_voice', type=str)
@click.argument('message', type=str, default=DEFAULT_MESSAGE)
@click.option('--path_to_result', type=str, default='./result.wav')
@click.option('--play_result', type=bool, default=False)
def clone_voice(path_to_voice, message, path_to_result, play_result):
    main_clone_voice(path_to_voice, message, path_to_result, play_result)


cli.add_command(clone_voice)


if __name__ == '__main__':
    cli()
