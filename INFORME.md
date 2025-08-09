# Tarea 3 — Comunicación colectiva y medición de latencia (mpi4py)

**Integrantes:** Sebastián Alvarado y Tanisha Miranda  
**Profesor:** Johansen Villalobos Cubillo  
**Curso:** 2025-II Computación paralela y distribuida

## 1) Resumen
Trabajamos en dos cosas: (A) operaciones **colectivas** con mpi4py para calcular mínimo, máximo y promedio global; y (B) medición de **latencia punto a punto** con un ping–pong Send/Recv. Corrimos todo en la misma máquina (shared-memory) y dejamos los comandos y resultados listos para reproducir.

## 2) Parte A — Colectivas (Bcast/Scatter/Reduce)
Lo que hicimos fue: el **rank 0** genera un arreglo grande de `N` elementos (modo float o int), hacemos **Bcast** de la configuración, repartimos el arreglo en partes iguales con **Scatter** y cada proceso calcula min, max y **suma local**. En vez de promediar locales (que se rompe si los tamaños fueran distintos), **reducimos la suma total** con `MPI.SUM` y sacamos el promedio global como `SUMA/N`. También reducimos `MIN` y `MAX`.

**Resultado ejemplo:** con `N=1,000,000`, `-np 4` y `float64`, obtuvimos: min ≈ 0.00011, max ≈ 99.99999, promedio ≈ 49.98949. Es lo esperable con datos uniformes en [0,100].

## 3) Parte B — Latencia punto a punto (Send/Recv)
Para medir latencia, usamos **dos procesos**. El rank 0 envía un mensaje y espera la respuesta; el rank 1 recibe y devuelve. Hicimos un **warmup** corto y después medimos `iters` veces. El tiempo por iteración es el **RTT** (ida y vuelta); la latencia unidireccional la aproximamos como `RTT/2`. Probamos con tamaños 1 B, 4 KiB, 64 KiB y 1 MiB.

**Nuestros datos:**

| size_bytes | RTT (µs) | One-way (µs) |
|-----------:|---------:|-------------:|
| 1          | 1.27     | 0.64         |
| 4,096      | 3.32     | 1.66         |
| 65,536     | 23.78    | 11.89        |
| 1,048,576  | 90.98    | 45.49        |

Las figuras `latency_rtt.png` y `latency_bw.png` muestran dos cosas: (1) la zona **dominada por latencia** en mensajes pequeños (RTT en el orden de microsegundos, casi constante), y (2) a medida que el mensaje crece, la métrica se desplaza a estar **limitada por ancho de banda** y el throughput sube hasta estabilizarse.

## 4) Interpretación breve
- En mensajes muy cortos, el costo fijo del stack (llamadas MPI, sincronización, copias) manda, por eso la curva de RTT arranca casi plana. Ese ≈1–2 µs es nuestra **latencia base**.
- Cuando los mensajes crecen (≥ decenas de KiB), lo que manda es la **velocidad de copia** entre espacios de memoria/procesos. Por eso el throughput sube y luego tiende a estabilizarse.
- Como corrimos en la misma máquina, estos números reflejan más bien límites de **memoria compartida** y la implementación local de MPI. En un cluster real, la red (latencia y ancho de banda) cambiaría bastante los valores.

