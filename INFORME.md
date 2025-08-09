# Tarea 3 — Comunicación colectiva y medición de latencia (mpi4py)

**Integrantes:** Sebastián Alvarado y Tanisha Miranda  
**Profesor:** Johansen Villalobos Cubillo  
**Curso:** 2025-II Computación paralela y distribuida

---

## 1) Objetivo y alcance
El trabajo tiene dos metas claras: (A) aplicar **operaciones colectivas** de MPI con `mpi4py` para calcular **mínimo, máximo y promedio global** de un arreglo grande; y (B) **medir la latencia** punto a punto con un esquema ping–pong (`Send/Recv`). Todo se ejecutó en una misma máquina (memoria compartida) y dejamos el proceso **reproducible** con comandos y artefactos de salida.

---

## 2) Metodología (visión general)
- **Parte A (colectivas):** el **rank 0** genera los datos y difunde la configuración; luego repartimos el arreglo con `Scatter`, cada proceso calcula sus estadísticas **locales** y el **root** consolida con `Reduce` usando `MPI.MIN`, `MPI.MAX` y `MPI.SUM`. El promedio global lo calculamos como `SUMA / N` (evita sesgos si los trozos no fueran idénticos).
- **Parte B (latencia):** dos procesos en ping–pong. Hicimos **warmup** para estabilizar, medimos `iters` ciclos y reportamos **RTT** (ida-y-vuelta) y una aproximación **unidireccional = RTT/2**. Evaluamos tamaños 1 B, 4 KiB, 64 KiB y 1 MiB, con opción de `Barrier` antes de cada medición.

---

## 3) Parte A — Colectivas (Bcast / Scatter / Reduce)

**Diseño y ejecución (resumen):**
1. Rank 0 crea un arreglo de tamaño `N` con valores uniformes en [0, 100] (modo `float64`) o enteros (modo `int32`).
2. Difundimos configuración por **Bcast**.
3. Repartimos en partes iguales con **Scatter** (validamos que `N % procesos == 0`).
4. Cada proceso calcula **min, max y suma local**.
5. Reducimos en root: `MIN` con `MPI.MIN`, `MAX` con `MPI.MAX`, y **SUMA** con `MPI.SUM`; luego `promedio = SUMA / N`.

**Comando usado:**
```bash
mpirun -np 4 python estadisticas_mpi.py --n 1000000 --mode float --verbose
Resultado (real):

Mínimo: 0.000110

Máximo: 99.999989

Promedio: 49.989488

Estos valores son coherentes con un muestreo uniforme en [0, 100] (promedio ≈ 50).

4) Parte B — Latencia punto a punto (Send/Recv)
Diseño y ejecución (resumen):

Dos procesos exactos (-np 2).

Warmup para “romper” la latencia de arranque y estabilizar cachés.

Medición de iters ping–pong por tamaño; el RTT promedio sale del tiempo total/iteraciones.

Reportamos one-way como RTT/2 (aproximación estándar para ping–pong simétrico).

Comando usado:

bash
Copy
Edit
mpirun -np 2 python latencia_mpi.py --iters 15000 --sizes 1,4096,65536,1048576 --barrier --csv results/latencias.csv
Resultados (reales):

size_bytes	RTT (µs)	One-way (µs)
1	1.27	0.64
4,096	3.32	1.66
65,536	23.78	11.89
1,048,576	90.98	45.49

Artefactos: results/latencias.csv, results/latency_rtt.png (RTT vs tamaño) y results/latency_bw.png (throughput one-way vs tamaño).
Estimación de throughput one-way para 1 MiB: ≈ 23 GB/s (1,048,576 bytes / 45.49 µs).

5) Análisis e interpretación
Zona dominada por latencia (mensajes pequeños): el RTT se mantiene casi plano (~1–2 µs). Ese es el costo fijo del stack MPI (llamadas, sincronización y copias mínimas).

Transición a zona dominada por ancho de banda (mensajes grandes): al crecer el tamaño, el tiempo depende de la velocidad efectiva de copia y el throughput sube hasta estabilizarse.

Contexto de ejecución: corrimos en una sola máquina (memoria compartida). En un clúster real, la red (latencia y ancho de banda) modificaría los números, especialmente para tamaños medianos y grandes.

6) Decisiones técnicas (y por qué)
Promedio global por SUMA/N en lugar de promediar promedios locales → robusto si los trozos no son iguales.

Validación previa de N % procesos == 0 → evita Scatter inconsistente y errores silenciosos.

Warmup + Barrier opcional → reducimos ruido de arranque y medimos en condiciones comparables.

Logs solo en root para resultados finales → salida limpia y fácil de revisar.

Tipos de dato cuidados (float64 / suma en int64) → precisión y sin overflow.

7) Reproducibilidad y trazabilidad
Entorno: macOS (portátil), Python 3.x, numpy, mpi4py, Open MPI instalado con Homebrew.
Comandos exactos usados:

bash
Copy
Edit
# Parte A
mpirun -np 4 python estadisticas_mpi.py --n 1000000 --mode float --verbose

# Parte B
mpirun -np 2 python latencia_mpi.py --iters 15000 --sizes 1,4096,65536,1048576 --barrier --csv results/latencias.csv
Artefactos entregados: scripts (estadisticas_mpi.py, latencia_mpi.py), README.md, results/latencias.csv y gráficas.
Fecha de corrida: 2025-08-08 (UTC-6, Costa Rica).

Con estos archivos, cualquier persona puede repetir la corrida y verificar los mismos resultados.

8) Conclusión
Cumplimos los objetivos: implementamos correctamente las colectivas y medimos la latencia con un diseño claro, validado y reproducible. Los resultados son consistentes con la teoría (latencia base en mensajes pequeños y saturación por ancho de banda en grandes). Dejamos el proyecto en estado de entrega, con documentación, datos y gráficos que respaldan el trabajo.

Copy
Edit
