import pytest


class Helpers:
    @staticmethod
    def is_module_equal(module1, module2):
        """Tests whether two PyTorch modules have the same structure.

        Args:
            module1 (torch.nn.Module): The first module to test.
            module2 (torch.nn.Module): The second module to test.

        Returns:
            bool: True if the modules have the same structure, False otherwise.
        """
        if type(module1) != type(module2):
            return False

        # Check if the modules have the same number of parameters
        if len(list(module1.parameters())) != len(list(module2.parameters())):
            return False

        # Check if the modules have the same number of children modules
        if len(list(module1.children())) != len(list(module2.children())):
            return False

        # Recursively check the structure of the children modules
        for child1, child2 in zip(module1.children(), module2.children()):
            if not Helpers.is_module_equal(child1, child2):
                return False

        return True


@pytest.fixture
def helpers():
    return Helpers
