function MLR(X,Y,Y_index,str,space)
Y_at_desired_index = Y(:,Y_index);
fprintf('Linear Regression Model for S11 %s part at frequency %d in the %s space.\n', str, Y_index-1,space)
mdl=fitlm(X,Y_at_desired_index)
residuals = mdl.Residuals.Raw;
mse = mean(residuals.^2);
disp(['Mean Squared Error (MSE): ', num2str(mse)]);
fprintf('\n')
Y_pred_at_desired_index = mdl.Fitted;
figure;
hold on
yline(0,'r');
scatter(Y_pred_at_desired_index,residuals);
xlabel("predicted values")
ylabel("residual")
title(sprintf('Residual Plot at frequency %d for %s part in the %s space.', Y_index-1,str,space))
hold off
figure;
grid on
scatter(Y_at_desired_index,Y_pred_at_desired_index)
hold on
min_val = min([Y_at_desired_index; Y_pred_at_desired_index]); 
max_val = max([Y_at_desired_index; Y_pred_at_desired_index]); 
plot([min_val, max_val], [min_val, max_val])
xlabel("observed values")
ylabel("predicted values")
title(sprintf('Observed vs. Predicted at frequency %d for %s part in the %s space.', Y_index-1,str,space))
hold off
end

