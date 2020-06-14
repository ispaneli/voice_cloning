import pytest
from main import clone_voice
import os
from pathlib import Path


def delete_result(path_to_result: str = './result.wav'):
    path_to_result = Path(path_to_result)

    if path_to_result.exists():
        os.remove(path_to_result)


def test_all_good_params():
    delete_result()
    clone_voice('./test_voice.wav', "Hello, world!", './result.wav')
    result_size = os.path.getsize(Path('result.wav'))

    assert 100_000 < result_size < 300_000, f"The result weight is out of the allowed range: {result_size}"


@pytest.mark.parametrize('path_to_voice', (True, None, 19.99, ['hello', 'world']))
def test_wrong_type_path_to_voice(path_to_voice):
    try:

        clone_voice(path_to_voice, "Hello, world!", './result.wav')
        assert False, f"TypeError for {path_to_voice} not invoked"
    except TypeError:
        assert True


@pytest.mark.parametrize('message', (True, None, 19.99, ['hello', 'world']))
def test_wrong_type_message(message):
    try:
        delete_result()
        clone_voice('./test_voice.wav', message, './result.wav')
        assert False, f"TypeError for {message} not invoked"
    except TypeError:
        assert True


@pytest.mark.parametrize('path_to_result', (True, None, 19.99, ['hello', 'world']))
def test_wrong_type_path_to_result(path_to_result):
    try:
        delete_result()
        clone_voice('./test_voice.wav', "Hello, world!", path_to_result)
        assert False, f"TypeError for {path_to_result} not invoked"
    except TypeError:
        assert True


@pytest.mark.parametrize('path_to_voice', ('../test_voice.wav', './test_voice.wave', './test_voice.wav/'))
def test_wrong_value_path_to_voice(path_to_voice):
    try:
        delete_result()
        clone_voice(path_to_voice, "Hello, world!", './result.wav')
        assert False, f"ValueError for {path_to_voice} not invoked"
    except ValueError:
        assert True


@pytest.mark.parametrize('message', ("русское msg", "1 eng 2 msg 3", "%#$"))
def test_wrong_value_message(message):
    try:
        delete_result()
        clone_voice('./test_voice.wav', message, './result.wav')
        assert False, f"ValueError for {message} not invoked"
    except ValueError:
        assert True


@pytest.mark.parametrize('path_to_result', ('../result.wav', './result.wave', './result.wav/'))
def test_wrong_value_path_to_result(path_to_result):
    try:
        delete_result()
        clone_voice('./test_voice.wav', "Hello, world!", path_to_result)
        assert False, f"ValueError for {path_to_result} not invoked"
    except ValueError:
        assert True

