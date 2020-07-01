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
@click.option('--path_to_voice', type=str)
@click.option('--message', type=str, default=DEFAULT_MESSAGE)
@click.option('--path_to_result', type=str, default='result.wav')
@click.option('--play_result', type=bool, default=False)
def clone_voice(path_to_voice, message, path_to_result, play_result):
    """
    Converting a fragment of a person's voice and a message into a voice message voiced by this voice.

    EXAMPLE OF USE:
    python3 cli.py clone-voice --path_to_voice='voice_for_test.wav' --message='My text' --path_to_result='result.wav' --play_result=True

    :param path_to_voice: Path to the file with the example of a human voice.
    :param message: The text that the neural network should voice.
    :param path_to_result: The path where the result of the program should be saved.
    :param play_result: Voice the result of the program execution or not.
    :return: None
    """
    main_clone_voice(path_to_voice, message, path_to_result, play_result)


cli.add_command(clone_voice)


if __name__ == '__main__':
    cli()
