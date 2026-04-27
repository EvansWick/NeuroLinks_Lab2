% 1. Формування сигналу зі зміною частоти
t1 = 0:0.01:3; 
x1 = sin(2*pi*t1);
t2 = 3.025:0.025:5; 
x2 = sin(4*pi*t2);
x_total = [x1 x2];
X = con2seq(x_total); 
% ВИПРАВЛЕННЯ: Формуємо цільовий вихід y(t) = 3x(t-1) + 1
% Зміщуємо вхідний масив вправо на 1 елемент, додаючи 0 на початок
x_delayed = [0, x_total(1:end-1)]; 
y_total = 3 * x_delayed + 1;
T = con2seq(y_total); 
% 2. Створення адаптивної лінійної мережі (2 блоки затримки)
net = linearlayer(1:2, 0.001); 
% 3. Підготовка даних та навчання
[Xs, Xi, Ai, Ts] = preparets(net, X, T); 
net.trainParam.epochs = 2000;
net.trainParam.goal = 1e-6;
net = train(net, Xs, Ts, Xi, Ai); 
% 4. Моделювання
Y = sim(net, Xs, Xi); 
Y_mat = cat(2, Y{:}); 
Ts_mat = cat(2, Ts{:});
error = Ts_mat - Y_mat;
% ГРАФІК 1: Порівняння цілі та виходу (той самий "ісходний" графік)
figure;
% 1. Графік ВХІДНОГО сигналу x(t) - "чисті дані"
subplot(3,1,1);
plot(x_total, 'k', 'LineWidth', 1);
title('Вхідний сигнал x(t) (зміна частоти та дискретизації)');
ylabel('x(t)');
grid on;
% 2. Порівняння ЦІЛІ та ВИХОДУ НМ - "що мало бути" vs "що отримали"
subplot(3,1,2);
plot(Ts_mat, 'b', 'LineWidth', 1.5); hold on;
plot(Y_mat, 'r--', 'LineWidth', 1.2);
title('Порівняння цілі y(t) = 3x(t-1)+1 та виходу мережі');
legend('Ціль', 'Вихід НМ');
ylabel('Значення');
grid on;
% 3. Графік помилки
subplot(3,1,3);
plot(error, 'm');
title('Графік помилки (різниця між ціллю та виходом)');
xlabel('Номер відліку');
ylabel('Помилка');
grid on;