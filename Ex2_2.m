% 4. Формування даних вручну (заміна prprob)
% Створюємо ідеальну одиничну матрицю для цілей (26 класів) 
targets = eye(26); 
% Створюємо матрицю алфавіту (35 ознак на 26 літер) 
% Оскільки функція prprob не знайдена, ми заповнимо її випадковими бітами, 
% щоб ти міг виконати логіку навчання та аналіз шуму.
alphabet = round(rand(35, 26)); 
% Якщо хочеш додати "реальності", ось як виглядає літера "G" з твоєї методички[cite: 702]:
G_pattern = [1 0 0 0 1; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 1 1; 1 0 0 0 1; 0 1 1 1 0; 0 0 0 0 0]';
alphabet(:, 7) = G_pattern(:); % Записуємо її на 7-му позицію
% 5. Створення мережі (31 нейрон у прихованому шарі)
net = feedforwardnet(31, 'trainlm'); 
net.divideFcn = ''; % Використовуємо всі дані для навчання
% Навчання
net.trainParam.goal = 1e-6;
net = train(net, alphabet, targets);
disp('Мережа успішно навчена на замінних даних!');
% Дослідження шуму
noise_levels = 0:0.05:0.5;
mean_errors = zeros(size(noise_levels));
for i = 1:length(noise_levels)
    std_dev = noise_levels(i);
    total_error = 0;
    
    for char_idx = 1:26
        for trial = 1:10
            % Додавання шуму
            noisy_input = alphabet(:, char_idx) + std_dev * randn(35, 1);
            % Моделювання виходу
            y_out = net(noisy_input);
            % Обчислення евклідової норми помилки
            total_error = total_error + norm(targets(:, char_idx) - y_out);
        end
    end
    % Обчислення середньої сумарної помилки для рівня шуму
    mean_errors(i) = total_error / (26 * 10);
end
% Побудова графіка
figure;
plot(noise_levels, mean_errors, 'r-s', 'LineWidth', 2);
grid on;
xlabel('Рівень шуму (std dev)');
ylabel('Середня помилка (Euclidean norm)');
title('Стійкість розпізнавання алфавіту до зашумлення');