# Tarea 3 — Comunicación colectiva y medición de latencia (mpi4py)

**Integrantes:** Sebastián Alvarado y Tanisha Miranda  
**Profesor:** Johansen Villalobos Cubillo  
**Curso:** 2025-II Computación paralela y distribuida

---

## 1) Objetivo y alcance
El trabajo tiene dos metas:  
(A) aplicar **operaciones colectivas** de MPI con `mpi4py` para calcular **mínimo, máximo y promedio global** de un arreglo grande;  
(B) **medir la latencia** punto a punto con un esquema ping–pong (`Send/Recv`).  
Ejecutamos todo en una misma máquina (memoria compartida) y dejamos el proceso **reproducible** con comandos y artefactos.

---

## 2) Metodología (visión general)
- **Parte A (colectivas):** el **rank 0** genera datos y difunde la configuración con `Bcast`. Repartimos con `Scatter`. Cada proceso calcula **min, max y suma local**. En el **root** reducimos con `MPI.MIN`, `MPI.MAX` y `MPI.SUM`. El **promedio global** lo calculamos como `SUMA / N`.
- **Parte B (latencia):** dos procesos en **ping–pong**. Hicimos **warmup** para estabilizar, medimos `iters` y reportamos **RTT** (ida y vuelta) y **one-way ≈ RTT/2**. Probamos tamaños 1 B, 4 KiB, 64 KiB y 1 MiB, con `Barrier` opcional antes de cada medición.

---

## 3) Parte A — Colectivas (Bcast / Scatter / Reduce)

### Diseño y ejecución (resumen)
1. **Rank 0** crea un arreglo de tamaño `N` con valores uniformes en `[0, 100]` (modo `float64`) o enteros (`int32`).  
2. Difundimos la **configuración** con **`Bcast`**.  
3. Repartimos en partes **iguales** con **`Scatter`** *(validamos que `N % procesos == 0`)*.  
4. Cada proceso calcula **mínimo**, **máximo** y **suma local** sobre su trozo.  
5. En **root** reducimos: `MIN` con `MPI.MIN`, `MAX` con `MPI.MAX` y **SUMA** con `MPI.SUM`; luego el **promedio global** = `SUMA / N`.

### Comando usado
```bash
mpirun -np 4 python estadisticas_mpi.py --n 1000000 --mode float --verbose
```

### Resultado (real)
- **Mínimo:** 0.000110  
- **Máximo:** 99.999989  
- **Promedio:** 49.989488  

> Coherente con muestreo uniforme en `[0, 100]` (promedio ≈ 50).

### Notas clave
- Promedio global vía **`SUMA / N`** (no “promedio de promedios”) → robusto si los trozos difieren.  
- Validamos **`N % procesos == 0`** antes de `Scatter`.  
- Tipos de dato: generación en `float64`/`int32`; **suma** en `float64`/`int64` para evitar pérdida u overflow.  
- **Salida limpia:** el resultado final lo imprime solo el **root**; `--verbose` muestra detalle por proceso.

---

## 4) Parte B — Latencia punto a punto (Send/Recv)

### Diseño y ejecución (resumen)
- **Dos procesos exactos** (`-np 2`).  
- **Warmup** para “romper” latencia de arranque y estabilizar cachés.  
- Medimos `iters` ping–pong por tamaño; **RTT** promedio = tiempo total / iteraciones.  
- Reportamos **one-way** como `RTT/2` (aprox. estándar para ping–pong simétrico).  

### Comando usado
```bash
mpirun -np 2 python latencia_mpi.py --iters 15000 --sizes 1,4096,65536,1048576 --barrier --csv results/latencias.csv
```

### Resultados (reales)

| size_bytes | RTT (µs) | One-way (µs) |
|-----------:|---------:|-------------:|
| 1          | 1.27     | 0.64         |
| 4,096      | 3.32     | 1.66         |
| 65,536     | 23.78    | 11.89        |
| 1,048,576  | 90.98    | 45.49        |

**Artefactos:** `results/latencias.csv`, `results/latency_rtt.png` (RTT vs tamaño), `results/latency_bw.png` (throughput one-way vs tamaño).  
**Throughput one-way (1 MiB):** ≈ **23 GB/s** (`1,048,576 bytes / 45.49 µs`).

---

## 5) Análisis e interpretación
- **Zona dominada por latencia (mensajes pequeños):** RTT casi plano (~1–2 µs) por el **costo fijo** del stack MPI (llamadas, sincronización y copias mínimas).  
- **Transición a zona dominada por ancho de banda (mensajes grandes):** al crecer el tamaño, manda la **velocidad efectiva de copia**; el throughput sube y luego se estabiliza.  
- **Contexto:** ejecución en **una sola máquina** (memoria compartida). En un clúster real, la red (latencia y ancho de banda) cambiaría los números, sobre todo en tamaños medianos/grandes.

---

## 6) Decisiones técnicas (y por qué)
- **Promedio global por `SUMA/N`** → evita sesgos si los trozos no son iguales.  
- **Validación previa** de `N % procesos == 0` → evita `Scatter` inconsistente.  
- **Warmup + `Barrier` opcional** → menos ruido y comparaciones limpias.  
- **Logs solo en root** → salida clara y sin duplicados.  
- **Cuidado de tipos** (`float64`; suma en `int64` para enteros) → precisión y sin overflow.

---

## 7) Reproducibilidad y trazabilidad
**Entorno:** macOS (portátil), Python 3.x, `numpy`, `mpi4py`, Open MPI (Homebrew).  

**Comandos exactos usados**
```bash
# Parte A
mpirun -np 4 python estadisticas_mpi.py --n 1000000 --mode float --verbose
```
```bash
# Parte B
mpirun -np 2 python latencia_mpi.py --iters 15000 --sizes 1,4096,65536,1048576 --barrier --csv results/latencias.csv
```

**Artefactos entregados:** `estadisticas_mpi.py`, `latencia_mpi.py`, `README.md`, `results/latencias.csv`, `results/latency_rtt.png`, `results/latency_bw.png`.  
**Fecha de corrida:** 2025-08-08 (UTC-6, Costa Rica).

> Con estos archivos, cualquier persona puede repetir la corrida y verificar los mismos resultados.

---

## 8) Conclusión
Cumplimos los objetivos: implementamos correctamente las **colectivas** y medimos la **latencia** con un diseño claro, validado y **reproducible**. Los resultados son consistentes con la teoría (latencia base en mensajes pequeños y saturación por ancho de banda en grandes). 
