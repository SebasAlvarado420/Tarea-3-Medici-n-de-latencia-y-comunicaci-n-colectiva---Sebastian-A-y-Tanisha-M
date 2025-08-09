# estadisticas_mpi.py
# ------------------------------------------------------------
# ¿Qué hace?
#   Calculamos min, max y promedio GLOBALES de un arreglo gigante usando
#   operaciones colectivas de MPI: Bcast, Scatter y Reduce (con mpi4py).
#
# ¿Cómo lo usamos?
#   - Definimos N (tamaño total) y el modo (float o int) por CLI.
#   - El root (rank 0) genera los datos y los transmite.
#   - Cada proceso calcula min/max/suma local.
#   - En root reducimos MIN, MAX y la SUMA total; el promedio global = SUMA/N.
#
# Ejemplos de ejecución:
#   mpirun -np 4 python estadisticas_mpi.py --n 1000000 --mode float
#   mpirun -np 8 python estadisticas_mpi.py --n 2000000 --mode int --seed 123 --verbose
# ------------------------------------------------------------

from mpi4py import MPI
import numpy as np
import argparse
import sys

# ---------------------------
# Utilidades y validaciones
# ---------------------------

def parse_args():
    """Parseo rápido y flexible. Todos los ranks lo corren, pero el root manda la versión 'oficial' por Bcast."""
    # Armamos los flags que vamos a aceptar por CLI (tamaño, tipo, semilla, verbose)
    parser = argparse.ArgumentParser(
        description="MPI: min/max/promedio global con Bcast/Scatter/Reduce (mpi4py)."
    )
    # --n: tamaño total del arreglo que queremos procesar
    parser.add_argument("--n", type=int, default=1_000_000,
                        help="Tamaño total del arreglo (ej: 1000000).")
    # --mode: elegimos float (decimales) o int (enteros) para los datos
    parser.add_argument("--mode", choices=["float", "int"], default="float",
                        help="Tipo de datos: 'float' (0..100 con decimales) o 'int' (0..100 enteros).")
    # --seed: si queremos reproducibilidad, pasamos una semilla
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para el RNG (si querés reproducibilidad).")
    # --verbose: imprime detalles de cada proceso (útil cuando depuramos)
    parser.add_argument("--verbose", action="store_true",
                        help="Imprime detalles de cada proceso (útil para depurar).")
    args = parser.parse_args()
    # Devolvemos un diccionario simple para transmitirlo por Bcast
    return {
        "N": int(args.n),
        "mode": args.mode,
        "seed": args.seed,
        "verbose": bool(args.verbose),
    }

def validate_config(cfg, size, rank, comm):
    """
    Validamos todo lo que puede explotar feo:
    - N > 0
    - N % size == 0 (para repartir exacto)
    - N >= size (evitar subarreglos de tamaño 0)
    Si algo falla, lo decimos y abortamos para que no quede nada zombie.
    """
    # Acá vamos a ir marcando si la config está bien o mal y juntando mensajes
    ok = True
    msgs = []

    N = cfg["N"]

    if N <= 0:
        ok = False
        msgs.append(f"N debe ser > 0 (llegó {N}).")

    if N < size:
        ok = False
        msgs.append(f"N ({N}) es menor que el número de procesos ({size}). Subarreglos de tamaño 0 no sirven.")

    if N % size != 0:
        ok = False
        msgs.append(f"N ({N}) no es divisible entre procesos ({size}).")

    if cfg["mode"] not in ("float", "int"):
        ok = False
        msgs.append(f"mode inválido: {cfg['mode']}")

    # Root junta el mensaje y decide
    if rank == 0 and not ok:
        print("\n[ERROR] Configuración inválida:")
        for m in msgs:
            print(f" - {m}")
        print("Sugerencia: ajustá --n para que sea múltiplo de -np, y mayor o igual a -np.")
    ok = comm.bcast(ok, root=0)
    if not ok:
        comm.Barrier()
        MPI.COMM_WORLD.Abort(1)

def init_data_on_root(cfg, rank, size):
    """
    Solo el root crea el arreglo gigante. El resto deja data=None.
    - Si mode=float: uniform(0,100) en float64
    - Si mode=int: integers 0..100 en int32 (sumamos en int64 para ir sobrados)
    """
    # Si NO somos root, no generamos datos, devolvemos None
    if rank != 0:
        return None

    # Tomamos N y preparamos el generador con o sin semilla
    N = cfg["N"]
    rng = np.random.default_rng(cfg["seed"])

    # Según el modo, creamos floats o enteros en el rango 0..100
    if cfg["mode"] == "float":
        data = rng.uniform(0.0, 100.0, size=N).astype(np.float64, copy=False)
    else:
        data = rng.integers(0, 101, size=N, dtype=np.int32)

    # Dejamos un log simple en root para saber qué hicimos
    print(f"Proceso raíz: Arreglo inicializado con {N} elementos ({data.dtype}).")
    return data

def scatter_data(comm, data, cfg, rank, size):
    """
    Repartimos el arreglo en partes iguales. Cada proceso recibe su 'rebanada'.
    Tip: Creamos el recvbuf del tipo correcto según mode.
    """
    # Calculamos cuántos elementos le toca a cada proceso
    N = cfg["N"]
    chunk = N // size

    # Armamos el buffer receptor con el dtype correcto
    if cfg["mode"] == "float":
        subarray = np.empty(chunk, dtype=np.float64)
    else:
        subarray = np.empty(chunk, dtype=np.int32)

    # Scatter: root manda 'data', los demás pasan None sin drama
    comm.Scatter([data, MPI.DOUBLE] if (rank == 0 and cfg["mode"] == "float") else
                 ([data, MPI.INT] if (rank == 0 and cfg["mode"] == "int") else None),
                 subarray, root=0)

    # Si pedimos verbosity, mostramos un pequeño resumen de lo recibido
    if cfg["verbose"]:
        print(f"Proceso {rank}: Subarreglo recibido con {subarray.size} elementos ({subarray.dtype})")

    return subarray

def local_stats(subarray, cfg, rank):
    """
    Sacamos min, max y suma local. Para el promedio global no voy a promediar aquí
    (promedio de promedios puede ser engañoso si cambian tamaños). Mejor reducimos SUMA.
    """
    if cfg["mode"] == "float":
        local_sum = float(np.sum(subarray, dtype=np.float64))
        local_min = float(np.min(subarray))
        local_max = float(np.max(subarray))
    else:
        local_sum = int(np.sum(subarray, dtype=np.int64))   
        local_min = int(np.min(subarray))
        local_max = int(np.max(subarray))

    # Si estamos depurando, mostramos un “preview” del promedio local
    if cfg["verbose"]:
        if isinstance(local_sum, float):
            avg_preview = local_sum / subarray.size if subarray.size else float("nan")
            print(f"Proceso {rank}: min={local_min:.3f}, max={local_max:.3f}, avg_local≈{avg_preview:.6f}")
        else:
            avg_preview = local_sum / subarray.size if subarray.size else float("nan")
            print(f"Proceso {rank}: min={local_min}, max={local_max}, avg_local≈{avg_preview:.6f}")

    return local_min, local_max, local_sum

def global_stats(comm, local_min, local_max, local_sum, cfg, rank, size):
    """
    Juntamos todo:
    - MIN global con MPI.MIN
    - MAX global con MPI.MAX
    - SUMA global con MPI.SUM y recién ahí sacamos el promedio en root
    """
    # Hacemos las reducciones: en root devuelven valor; en el resto, None
    gmin = comm.reduce(local_min, op=MPI.MIN, root=0)
    gmax = comm.reduce(local_max, op=MPI.MAX, root=0)
    gsum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # Solo root puede calcular el promedio final 
    if rank == 0:
        N = cfg["N"]
        gavg = float(gsum) / float(N)
        return gmin, gmax, gavg
    else:
        return None, None, None

# ---------------------------
# Main
# ---------------------------

def main():
    # Obtenemos el comunicador global y nuestros identificadores
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1) Parse local y broadcast de la config “oficial” 
    local_cfg = parse_args()
    cfg = comm.bcast(local_cfg if rank == 0 else None, root=0)

    # 2) Validaciones duras
    validate_config(cfg, size, rank, comm)

    # 3) Root inicializa datos, luego repartimos
    data = init_data_on_root(cfg, rank, size)
    subarray = scatter_data(comm, data, cfg, rank, size)

    local_min, local_max, local_sum = local_stats(subarray, cfg, rank)

    # 5) Reducimos al root y calculamos promedio global
    gmin, gmax, gavg = global_stats(comm, local_min, local_max, local_sum, cfg, rank, size)

    # 6) Imprimimos resultados finales SOLO en root 
    if rank == 0:
        if cfg["mode"] == "float":
            print(f"\nEstadísticas globales:")
            print(f"  Mínimo: {gmin:.6f}")
            print(f"  Máximo: {gmax:.6f}")
            print(f"  Promedio: {gavg:.6f}")
        else:
            print(f"\nEstadísticas globales:")
            print(f"  Mínimo: {gmin}")
            print(f"  Máximo: {gmax}")
            print(f"  Promedio: {gavg:.6f}")

if __name__ == "__main__":
    # Mini protección por si alguien lo ejecuta sin mpirun
    if MPI.COMM_WORLD.Get_size() == 1:
        print("[ADVERTENCIA] Estás corriendo con 1 proceso. Probá con mpirun -np 4 (o más) para ver las colectivas funcionar.")
    main()
