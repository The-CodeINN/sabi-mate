# filepath: c:\Users\emije\Documents\sabi-mate\src\ai_companion\core\datetime_utils.py
from datetime import datetime
import pytz


def get_current_datetime(timezone_str="Africa/Lagos"):
    """
    Get the current date and time formatted in a user-friendly way based on the specified timezone.

    Args:
        timezone_str (str): The timezone string (e.g., 'Africa/Lagos', 'America/Los_Angeles', etc.)
                            Default is 'Africa/Lagos' which corresponds to Nigerian time.

    Returns:
        str: A formatted string with the current date and time in the specified timezone.
    """
    try:
        # Get the timezone
        timezone = pytz.timezone(timezone_str)

        # Get the current time in the specified timezone
        current_time = datetime.now(timezone)

        # Format it in a user-friendly way
        formatted_datetime = current_time.strftime("%A, %B %d, %Y at %I:%M %p")

        return formatted_datetime
    except Exception:
        # In case of any error, return a simple formatted date without timezone
        return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
