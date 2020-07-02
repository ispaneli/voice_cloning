from pathlib import Path
import os

import pytest

from main import clone_voice


def setup_function(function):
    """
    Deletes the file with the neural network result before each test.
    """
    path_to_result = Path('result.wav')

    if path_to_result.exists():
        os.remove(path_to_result)


def test_all_good_params():
    """
    Test 1.
    Checks the operation of the neural network with the correct input parameters.
    """
    clone_voice('voice_for_test.wav', "Hello, world!", 'result.wav')
    result_size = os.path.getsize(Path('result.wav'))

    assert 100_000 < result_size < 300_000, f"The result weight is out of the allowed range: {result_size}"


@pytest.mark.parametrize('path_to_voice', (True, None, 19.99, ['hello', 'world']))
def test_wrong_type_path_to_voice(path_to_voice):
    """
    Test 2.
    Checks the operation of the neural network if the file path type is incorrect.
    """
    try:
        clone_voice(path_to_voice, "Hello, world!", 'result.wav')
        assert False, f"TypeError for {path_to_voice} not invoked"
    except TypeError:
        assert True


@pytest.mark.parametrize('message', (True, None, 19.99, ['hello', 'world']))
def test_wrong_type_message(message):
    """
    Test 3.
    Checks the operation of the neural network if the message type is incorrect.
    """
    try:
        clone_voice('voice_for_test.wav', message, 'result.wav')
        assert False, f"TypeError for {message} not invoked"
    except TypeError:
        assert True


@pytest.mark.parametrize('path_to_result', (True, None, 19.99, ['hello', 'world']))
def test_wrong_type_path_to_result(path_to_result):
    """
    Test 4.
    Checks the operation of the neural network if the result path type is incorrect.
    """
    try:
        clone_voice('voice_for_test.wav', "Hello, world!", path_to_result)
        assert False, f"TypeError for {path_to_result} not invoked"
    except TypeError:
        assert True


@pytest.mark.parametrize('path_to_voice', ('../voice_for_test.wav', 'voice_for_test.wave', 'voice_example.wav'))
def test_wrong_value_path_to_voice(path_to_voice):
    """
    Test 5.
    Checks the operation of the neural network when the path to the file with the voice is incorrectly specified.
    """
    try:
        clone_voice(path_to_voice, "Hello, world!", 'result.wav')
        assert False, f"FileExistsError for {path_to_voice} not invoked"
    except FileExistsError:
        assert True


@pytest.mark.parametrize('message', ("русское msg", "1 eng 2 msg 3", "%#$"))
def test_wrong_value_message(message):
    """
    Test 6.
    Checks the operation of the neural network when the message is incorrectly specified.
    """
    try:
        clone_voice('voice_for_test.wav', message, 'result.wav')
        assert False, f"ValueError for {message} not invoked"
    except ValueError:
        assert True


@pytest.mark.parametrize('path_to_result', ('../result.wav', 'result.wave'))
def test_wrong_value_path_to_result(path_to_result):
    """
    Test 7.
    Checks the operation of the neural network when the path to the result is incorrectly specified.
    """
    try:
        clone_voice('voice_for_test.wav', "Hello, world!", path_to_result)
        assert False, f"FileExistsError for {path_to_result} not invoked"
    except FileExistsError:
        assert True
