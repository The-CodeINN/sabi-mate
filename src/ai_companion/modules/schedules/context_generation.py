from datetime import datetime, time
from typing import Dict, Optional

from ai_companion.core.schedules import (
    SUNDAY_SCHEDULE,
    MONDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
    THURSDAY_SCHEDULE,
    FRIDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
)


class ScheduleContextGenerator:
    """Generates context about current activity based on schedules."""

    SCHEDULES = {
        0: MONDAY_SCHEDULE,
        1: TUESDAY_SCHEDULE,
        2: WEDNESDAY_SCHEDULE,
        3: THURSDAY_SCHEDULE,
        4: FRIDAY_SCHEDULE,
        5: SATURDAY_SCHEDULE,
        6: SUNDAY_SCHEDULE,
    }

    @staticmethod
    def _parse_time_range(time_range: str) -> tuple[time, time]:
        """Parse a time range string into start and end times."""
        start_str, end_str = time_range.split("-")
        start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
        end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
        return start_time, end_time

    @classmethod
    def get_current_activity(cls) -> Optional[str]:
        """Retrieve the current activity based on the current time and schedules.

        Returns:
            str: Current activity or None if no activity is found.
        """
        current_datetime = datetime.now()
        current_time = current_datetime.time()
        current_day = current_datetime.weekday()

        # Get the schedule for the current day
        schedule = cls.SCHEDULES.get(current_day, {})

        # Find matching activity
        for time_range, activity in schedule.items():
            start_time, end_time = cls._parse_time_range(time_range)

            # Handle overnight activities e.g ()
            if start_time > end_time:
                if current_time >= start_time or current_time <= end_time:
                    return activity

            else:
                if start_time <= current_time <= end_time:
                    return activity
        return None

    @classmethod
    def get_schedule_for_day(cls, day: int) -> Dict[str, str]:
        """Get the complete schedule for a specific day

        Args:
            day: Day of week as integer (0=Monday, 6=Sunday)

        Return:
            Dict[str, str]: Schedule for the specified day
        """
        return cls.SCHEDULES.get(day, {})
