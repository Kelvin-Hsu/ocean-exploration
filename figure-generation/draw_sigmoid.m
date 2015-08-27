%% Drawing a Sigmoid

function draw_sigmoid()
    
    close all;
    
    set(0, 'DefaultAxesFontName', 'Times New Roman')
    set(0, 'DefaultAxesFontSize', 20)

    x = -10:0.1:10;
    y1 = normcdf(x);
    y2 = logistic(x);
    

    figure();
    hold on;
    plot(x, y1, 'Color', [0, 0.8, 1.0], 'LineWidth', 2);
    plot(x, y2, 'Color', [0, 0.5, 1.0], 'LineWidth', 2);
    ylim([-0.5, 1.5]);
    title('Likelihood Responses');
    xlabel('\it f');
    ylabel('\it \sigma(f)');
    legend('Probit', 'Logistic')
    saveas(gcf, 'responses.png');
    print -depsc2 responses.eps
    
    x0 = -10;
    y0 = -0.5;
    
    f1bar = 0;
    f1sd = 0.2;
    f1 = linspace((f1bar - f1sd), (f1bar + f1sd), 50);
 
    f2bar = 1;
    f2sd = 0.4;
    f2 = linspace((f2bar - f2sd), (f2bar + f2sd), 50);
 
    f3bar = -1.8;
    f3sd = 0.2;
    f3 = linspace((f3bar - f3sd), (f3bar + f3sd), 50);
 
    f4bar = 7;
    f4sd = 0.8;
    f4 = linspace((f4bar - f4sd), (f4bar + f4sd), 50);
    
    function fill_compare(f, fbar)

        fill_under(f, normcdf(f), [0, 0.9, 0.1], y0);
        fill_left(f, normcdf(f), [0, 0.8, 1.0], x0);
        plot_join_axis(fbar, normcdf(fbar), x0, y0, 'k-.');
        L = @(f) linearised_normcdf(f, fbar);
        plot(x, L(x), '--', 'Color', [0.5, 0.5, 0], 'LineWidth', 0.5);

        % fill_left(f, L(f), [1, 0.2, 0], -10);
    end
    
    figure();
    hold on;
    plot(x, y1, 'Color', [1.0, 0.5, 0.0], 'LineWidth', 2);
    

    
    fill_compare(f1, f1bar);
    fill_compare(f2, f2bar);
    fill_compare(f3, f3bar);
    fill_compare(f4, f4bar);
    text(f1bar, y0 - 0.05, '$f^{\star}_{1}$', 'Interpreter', 'latex', 'FontSize', 16);
    text(f2bar, y0 - 0.05, '$f^{\star}_{2}$', 'Interpreter', 'latex', 'FontSize', 16);
    text(f3bar, y0 - 0.05, '$f^{\star}_{3}$', 'Interpreter', 'latex', 'FontSize', 16);
    text(f4bar, y0 - 0.05, '$f^{\star}_{4}$', 'Interpreter', 'latex', 'FontSize', 16);
    ylim([-0.5, 1.5]);
    title('Linearising the likelihood');
    xlabel('\it f');
    ylabel('\it \sigma(f)');
    


    saveas(gcf, 'linearisation.png');
    print -depsc2 linearisation.eps
end

function y = logistic(x)

    y = 1./(1 + exp(-x));
end

function y = linearised_normcdf(x, xbar)

    g0 = normpdf(xbar);
    y0 = normcdf(xbar);
    
    y = y0 + g0 * (x - xbar);
end

function plot_join_axis(x, y, x0, y0, varargin)

    plot([x, x], [y0, y], varargin{:});
    plot([x0, x], [y, y], varargin{:});
end

function fill_under(X, Y, C, y0)

    Xv = [X(1), X, X(end)];
    Yv = [y0, Y, y0];
    
    patch(Xv, Yv, C, 'EdgeColor', 'none');

    alpha(.2);
end

function fill_left(X, Y, C, x0)
    
    Xv = [x0, X, x0];
    Yv = [Y(1), Y, Y(end)];
    
    patch(Xv, Yv, C, 'EdgeColor', 'none');

    alpha(.2);
end