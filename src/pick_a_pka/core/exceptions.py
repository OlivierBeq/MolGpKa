class PickAPkaError(Exception):
    """Base exception for pick-a-pka package."""
    pass


class InvalidBackendError(PickAPkaError):
    """Raised when an unknown backend is requested."""
    pass


class InvalidMoleculeError(PickAPkaError):
    """Raised when a SMILES string or RDKit molecule is invalid."""
    pass


class ResourceNotFoundError(PickAPkaError):
    """Raised when model weights or reference files cannot be found."""
    pass


class QupkakeNotInstalledError(PickAPkaError):
    """Raised when Qupkake is not installed."""
    pass


class XTBError(PickAPkaError):
    """Base class for errors related to the xTB executable."""
    pass


class XTBNotFoundError(PickAPkaError):
    """Raised when the xTB executable is not found in the PATH."""
    pass


class XTBVersionError(PickAPkaError):
    """Raised when the installed xTB version does not meet requirements."""
    pass
