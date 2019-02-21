"""
Learning Google Style Guide
https://google.github.io/styleguide/pyguide.html


"""


class ConnectionThing(object):

    def __init__(self, user, passwd):
        self.user = user
        self.passwd = passwd


    def _find_next_open_port(self, current_port):
        next_port = current_port + 1

        if next_port  > 0:
            return next_port
        else:
            return False


    def connect_to_next_port(self, minimum):
        """Connects to the next available port.

        Args:
            minimum: A port value greater or equal to 1024.
        Raises:
            ValueError: If the minimum port specified is less than 1024.
            ConnectionError: If no available port is found.
        Returns:
            The new minimum port.
        """
        if minimum < 1024:
            raise ValueError(f'Minimum port must be at least 1024, not {minimum}')

        port = self._find_next_open_port(minimum)

        if not port:
            raise ConnectionError(f'Could not connect to service on {minimum} or higher.')

        assert port >= minimum, f'Unexpected port {port} when minimum was {minimum}.'

        return port


a_con = ConnectionThing('usr', 'pass')
a_con.connect_to_next_port(1028)









