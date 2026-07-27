import os
import re
import shutil
import subprocess

from ...core.exceptions import XTBNotFoundError, XTBVersionError, XTBError


def verify_xtb_working(xtb_path="xtb", expected_version="6.4.1") -> str:
    """Validate an explicitly-requested xTB override, checking for an exact
    version match (newer xTB releases change output QupKake's parser relies on)."""
    resolved_path = xtb_path if os.path.isabs(xtb_path) else shutil.which(xtb_path)

    if resolved_path is None or not os.path.exists(resolved_path):
        raise XTBNotFoundError(
            f"xTB executable not found: '{xtb_path}'. "
            "Omit xtb_path to use QupKake's bundled xTB 6.4.1 instead."
        )
    try:
        result = subprocess.run(
            [resolved_path, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        match = re.search(r"xtb\s+version\s+([\d.]+)", result.stdout)
        if not match:
            match = re.search(r"version\s+([\d.]+)", result.stdout)

        if match:
            installed_version_str = match.group(1)

            if installed_version_str != expected_version:
                raise XTBVersionError(
                    f"xTB version {installed_version_str} detected at '{resolved_path}', but "
                    f"QupKake requires exactly {expected_version} (newer versions changed xTB's "
                    "output format and break featurization). Omit xtb_path to use the bundled binary."
                )
        else:
            raise XTBVersionError("Could not verify xTB version.")

    except subprocess.CalledProcessError as e:
        raise XTBError(f"xTB was found, but failed to run. Error: {e.stderr}")
    except PermissionError:
        raise XTBError("xTB was found, but you do not have permission to execute it.")
    except ValueError:
        raise XTBVersionError("Could not parse xTB version string correctly.")

    return resolved_path
