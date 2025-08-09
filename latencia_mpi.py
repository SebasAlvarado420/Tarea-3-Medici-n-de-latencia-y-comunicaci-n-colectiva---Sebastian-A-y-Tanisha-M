# latencia_mpi.py
# ------------------------------------------------------------
# Medimos latencia punto a punto (ida y vuelta y ~unidireccional)
# Requiere EXACTAMENTE 2 procesos (-np 2)
# - rank 0: send -> recv (loop)
# - rank 1: recv -> send (loop)
# Imprimimos la latencia promedio por tamaño de mensaje.
# ------------------------------------------------------------

from mpi4py import MPI
import numpy as np
import argparse
import sys

def parse_args():
    # Armamos los argumentos que podemos pasar cuando corremos el script
    p = argparse.ArgumentParser(description="Medición de latencia MPI (Send/Recv).")
    # --iters: cuántas iteraciones medimos por cada tamaño de mensaje
    p.add_argument("--iters", type=int, default=10000, help="Iteraciones medidas por tamaño.")
    # --warmup: cuántas iteraciones hacemos antes para “calentar” y estabilizar
    p.add_argument("--warmup", type=int, default=200, help="Iteraciones de calentamiento (no se miden).")
    # --sizes: lista de tamaños en bytes separados por coma; por defecto: 1 B, 1 KiB, 1 MiB
    p.add_argument("--sizes", type=str, default="1,1024,1048576",
                   help="Tamaños de mensaje en bytes, separados por comas. Ej: 1,1024,1048576")
    # --barrier: si lo pasamos, sincronizamos con Barrier antes de cada medición para empezar parejo
    p.add_argument("--barrier", action="store_true", help="Sincroniza con Barrier antes de cada medición.")
    # --csv: si pasamos una ruta, guardamos resultados en un archivo CSV con columnas claras
    p.add_argument("--csv", type=str, default=None, help="Opcional: guardar resultados en CSV.")
    # Devolvemos el resultado parseado
    return p.parse_args()

def ensure_two_procs(comm):
    # Verificamos que el programa esté corriendo con exactamente 2 procesos
    size = comm.Get_size()
    if size != 2:
        if comm.Get_rank() == 0:
            print(f"[ERROR] Este script requiere -np 2 (recibió -np {size}).")
        MPI.COMM_WORLD.Abort(1)

def measure_size(comm, rank, msg_nbytes, iters, warmup, do_barrier):
    """
    Para un tamaño de mensaje específico (msg_nbytes), hacemos:
      - warmup: unas iteraciones que no medimos
      - medición: iters veces un ping-pong (RTT)
    Devolvemos dos números:
      - rtt_prom_s: tiempo promedio ida+vuelta por mensaje (en segundos)
      - one_way_prom_s: aproximamos la ida sola como RTT/2
    """
    tag = 77  # usamos un tag fijo para identificar estos mensajes
    buf = np.empty(msg_nbytes, dtype=np.uint8)

    # Calentamiento para estabilizar cachés, rutas y evitar medir arranques fríos
    for _ in range(warmup):
        if rank == 0:
            # rank 0 envía al 1 y luego espera la respuesta
            comm.Send([buf, MPI.BYTE], dest=1, tag=tag)
            comm.Recv([buf, MPI.BYTE], source=1, tag=tag)
        else:
            # rank 1 primero recibe y luego responde
            comm.Recv([buf, MPI.BYTE], source=0, tag=tag)
            comm.Send([buf, MPI.BYTE], dest=0, tag=tag)

    # Si el usuario pidió barrera, sincronizamos ambos procesos antes de medir
    if do_barrier:
        comm.Barrier()

    # Hacemos iters ping-pong y tomamos el tiempo total
    t0 = MPI.Wtime()
    for _ in range(iters):
        if rank == 0:
            comm.Send([buf, MPI.BYTE], dest=1, tag=tag)
            comm.Recv([buf, MPI.BYTE], source=1, tag=tag)
        else:
            comm.Recv([buf, MPI.BYTE], source=0, tag=tag)
            comm.Send([buf, MPI.BYTE], dest=0, tag=tag)
    t1 = MPI.Wtime()

    # El tiempo por iteración es un RTT (ida y vuelta de un solo mensaje)
    rtt_avg = (t1 - t0) / float(iters)
    one_way = rtt_avg / 2.0
    return rtt_avg, one_way

def main():
    # Obtenemos el comunicador global de MPI (todos los procesos)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ensure_two_procs(comm)

    # Leemos los argumentos que pasamos por línea de comandos
    args = parse_args()
    # Convertimos la lista de tamaños (string) en enteros
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    # Solo el root (rank 0) imprime el encabezado de la corrida
    if rank == 0:
        print(f"Medición de latencia con iters={args.iters}, warmup={args.warmup}, sizes={sizes}")

    # Acá vamos acumulando los resultados por si queremos escribir a archivo
    results = []  

    # Recorremos cada tamaño de mensaje y medimos
    for sz in sizes:
        if args.barrier:
            comm.Barrier()
        rtt, one_way = measure_size(comm, rank, sz, args.iters, args.warmup, args.barrier)

        # Solo el rank 0 imprime resultados para que no salga duplicado
        if rank == 0:
            rtt_us = rtt * 1e6
            one_way_us = one_way * 1e6
            print(f"Tamaño {sz:>8} B  |  RTT promedio: {rtt_us:9.2f} µs  |  Unidireccional ~ {one_way_us:9.2f} µs")
            results.append((sz, rtt, one_way))

    # Si nos dan una ruta en --csv, escribimos el archivo con columnas claras
    if rank == 0 and args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["size_bytes", "rtt_seconds", "one_way_seconds", "rtt_microseconds", "one_way_microseconds"])
            for sz, rtt, one_way in results:
                w.writerow([sz, rtt, one_way, rtt * 1e6, one_way * 1e6])
        print(f"\nCSV guardado en: {args.csv}")

if __name__ == "__main__":
    main()
