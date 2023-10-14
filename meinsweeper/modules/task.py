class Task:
    def __init__(self, name):
        self.name = name

    def configure(self):
        """Check if the machine is properly configured to run this task."""
        pass

    def run(self):
        """Run the task."""
        pass

    def complete(self):
        """Check if the task has completed successfully."""
        pass

    def log_progress(self):
        """Log the progress of the task."""
        pass