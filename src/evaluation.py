import json
import re


# These task-specific rules grade outputs; they do not select or train masks.
def correct(task, expected, response):
    try:
        if task == 'food':
            # Food item ordering does not matter, but names and quantities must match.
            canonical = lambda text: sorted((item['food'].strip().lower(), item['quantity'])
                                            for item in json.loads(text))
            return canonical(response.strip()) == canonical(expected)
        if task == 'addition':
            return bool(re.fullmatch(r'-?\d+', response.strip())) and int(response) == int(expected)
        # Appointment/policy answers must have exactly the expected JSON keys and types.
        value, target = json.loads(response), json.loads(expected)
        return type(value) is dict and set(value) == set(target) and all(
            type(value[key]) is type(target[key]) and value[key] == target[key] for key in target)
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
