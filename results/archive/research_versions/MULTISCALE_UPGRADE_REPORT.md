# گزارش نهایی ارتقای پروژه PINN به Multi-Scale بدون Fourier

## هدف
نسخه فعال پروژه برای مقایسه مستقیم **PINN ساده** و **Multi-Scale PINN بدون Fourier** آماده شده است. در مقایسه، seed، تعداد epoch، optimizer، learning rate، نقاط آموزش و روش ارزیابی برای هر مسئله یکسان هستند و فقط معماری شبکه تغییر می‌کند.

## معماری Multi-Scale فعال
ورودی خام `x` یا `(x,t)` مستقیماً وارد سه شاخه می‌شود و هیچ Fourier/sinusoidal encoding در مسیر فعال وجود ندارد.

- Low branch: عرض 8، عمق 1
- Mid branch: عرض 16، عمق 2
- High branch: عرض 24، عمق 3
- Feature dimension: 8
- Fusion dimension: 16
- Activation: Tanh
- Adaptive branch weighting: Softmax
- Fusion: Concatenation + Residual Fusion

تمام شاخه‌ها در forward واقعی مدل شرکت می‌کنند و مشتق‌های PINN از خروجی نهایی با PyTorch Autograd محاسبه می‌شوند.

## Baseline
Baseline همان PINN ساده موجود در `src/models/baseline_pinn.py` است:
- MLP تک‌مسیره
- hidden width = 32
- hidden layers = 3
- Tanh

## مسائل مقایسه‌شده
1. Exponential Decay ODE
2. Harmonic Oscillator ODE
3. Viscous Burgers Equation

برای Burgers:
- `x in [-1,1]`
- `t in [0,1]`
- `nu = 0.01/pi`
- IC = 100
- BC = 200
- PDE points = 2000
- epochs = 5000
- Adam, learning rate = 1e-3
- seed = 42

## نتیجه A/B فعلی
| Problem | Baseline Relative L2 | Multi-Scale Relative L2 | نتیجه |
|---|---:|---:|---|
| Decay | 4.862147e-05 | 4.366501e-04 | Baseline بهتر |
| Oscillator | 3.973069e-05 | 6.128259e-05 | Baseline بهتر |
| Burgers | 1.900072e-01 | 2.455006e-01 | Baseline بهتر |

نکته مهم: در Burgers، Multi-Scale به Final Loss پایین‌تری رسید (`9.0805e-03` در برابر `1.5875e-02`) اما Relative L2 بدتر شد. بنابراین کاهش training loss به تنهایی به معنی بهبود دقت حل نیست.

## خروجی مقایسه
- `results/tables/baseline_vs_multiscale_comparison.csv`
- `results/logs/baseline_vs_multiscale_comparison.txt`
- `results/figures/baseline_vs_multiscale_relative_l2.png`
- نمودارهای loss مقایسه‌ای برای هر سه مسئله
- نمودارهای prediction مقایسه‌ای برای ODEها
- raw loss CSV برای هر مدل/مسئله

## فایل‌های خالی قدیمی
تمام فایل‌های zero-byte باقی‌مانده از نسخه قبلی حذف شده‌اند. ساختار اصلی پوشه‌ها حفظ شده و فایل‌های غیرخالی قدیمی/آرشیوی حذف نشده‌اند.

## فایل‌های اصلی مرتبط
- `src/models/pinn.py`: مدل فعال Multi-Scale بدون Fourier
- `src/models/AMS_PINN.py`: پیاده‌سازی سه شاخه و adaptive fusion بدون Fourier
- `src/models/baseline_pinn.py`: PINN ساده baseline
- `experiments/compare_architectures.py`: اجرای مقایسه A/B
- `experiments/decay/run_decay.py`
- `experiments/oscillator/run_oscillator.py`
- `experiments/burgers/run_burgers.py`

## نتیجه علمی فعلی
معماری Multi-Scale بدون Fourier در فرم فعلی هنوز baseline را شکست نداده است. این نتیجه در پروژه حفظ شده تا مرحله بعدی بهینه‌سازی معماری بر اساس شواهد واقعی انجام شود، نه با حذف یا پنهان‌کردن نتایج نامطلوب.
