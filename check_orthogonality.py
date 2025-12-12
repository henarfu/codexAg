
import numpy as np

def check_orthogonality(name, matrix_path):
    try:
        matrix = np.load(matrix_path)
        print(f"Matrix {name} loaded from {matrix_path}")
        print(f"Shape of {name}: {matrix.shape}")

        is_square = matrix.shape[0] == matrix.shape[1]

        if is_square:
            # Check for full orthogonality
            identity = np.identity(matrix.shape[0])
            ortho_check = matrix @ matrix.T
            fro_norm_diff = np.linalg.norm(ortho_check - identity, 'fro')
            print(f"||{name} @ {name}.T - I|| (Frobenius norm): {fro_norm_diff}")
            if np.allclose(ortho_check, identity):
                print(f"{name} is an orthogonal matrix.")
            else:
                print(f"{name} is NOT an orthogonal matrix.")
        else:
            # Check for semi-orthogonality
            # Row orthogonality
            identity_rows = np.identity(matrix.shape[0])
            row_ortho_check = matrix @ matrix.T
            fro_norm_diff_rows = np.linalg.norm(row_ortho_check - identity_rows, 'fro')
            print(f"||{name} @ {name}.T - I|| (Frobenius norm): {fro_norm_diff_rows}")
            if np.allclose(row_ortho_check, identity_rows):
                print(f"{name} has orthogonal rows.")
            else:
                print(f"{name} does NOT have orthogonal rows.")

            # Column orthogonality
            identity_cols = np.identity(matrix.shape[1])
            col_ortho_check = matrix.T @ matrix
            fro_norm_diff_cols = np.linalg.norm(col_ortho_check - identity_cols, 'fro')
            print(f"||{name}.T @ {name} - I|| (Frobenius norm): {fro_norm_diff_cols}")
            if np.allclose(col_ortho_check, identity_cols):
                print(f"{name} has orthogonal columns.")
            else:
                print(f"{name} does NOT have orthogonal columns.")

    except FileNotFoundError:
        print(f"File not found: {matrix_path}")
    except Exception as e:
        print(f"An error occurred with matrix {name}: {e}")

if __name__ == "__main__":
    check_orthogonality('A', '/home/hdsp/RESULTS/AA.npy')
    print("-" * 30)
    check_orthogonality('B', '/home/hdsp/RESULTS/B_teacher02.npy')
