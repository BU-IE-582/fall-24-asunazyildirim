clc
clear al
close all

%PCA
input = readtable("C:\Users\asuna\OneDrive\Masaüstü\IE\hw1_files/hw1_input.csv", 'VariableNamingRule', 'preserve');
design_parameters = table2array(input);
design_parameters_scaled = zscore(design_parameters);
[coeff,score,latent,tsquared,explained,mu] = pca(design_parameters_scaled);
coeff
explained
cum_explained_var = cumsum(explained);
figure,
hold on
bar(explained)
plot(cum_explained_var, "o-")
xlabel("# of principal components")
ylabel("% of explained variance")
title("% of explained variance vs. # of principal components ")
legend("Individual", "Cumulative")
hold off

data_normal = rand(385,11);
[coeff2,score2,latent2,tsquared2,explained2,mu2] = pca(data_normal);
explained2
cum_explained_var2 = cumsum(explained2);
figure,
hold on
bar(explained2)
plot(cum_explained_var2, "o-")
xlabel("# of principal components")
ylabel("% of explained variance")
title("% of explained variance vs. # of principal components for randomly generated data ")
legend("Individual", "Cumulative")
hold off

limit= 80;
index = find(cum_explained_var >= limit, 1);
fprintf('The threshold of %.2f is first met with %d principal components with cumulative explained variance %.4f.\n\n',limit, index, cum_explained_var(index));
reduced_design_parameters = score(:,1:index);


%Regression
img_data = readtable("hw1_files/hw1_img.csv", 'VariableNamingRule', 'preserve');
s11_img_part_wfreq = table2array(img_data);
real_data = readtable("hw1_files/hw1_real.csv", 'VariableNamingRule', 'preserve');
s11_real_part_wfreq = table2array(real_data);

s11_freq = s11_img_part_wfreq(1,:);
s11_img_part = s11_img_part_wfreq(2:end,:);
s11_real_part = s11_real_part_wfreq(2:end,:);
s11_norm = (s11_img_part.^2 + s11_real_part.^2).^0.5;
s11_desibel = 20*log10(s11_norm);

figure;
plot(s11_freq,s11_norm)
xlabel("Frequency")
ylabel("S11 norm")
title("S11 norm vs Frequency")
figure;
plot(s11_freq,s11_desibel)
xlabel("Frequency")
ylabel("S11 in dB")
title("S11 in dB vs Frequency")

variance_s11_desibel = var(s11_desibel);
[sorted_variance ,index] = sort(variance_s11_desibel,"descend");
freq_index = index-1;
result_table = table(sorted_variance',freq_index','VariableNames',{'Variance','Frequency'});
disp(result_table);

freq_point_with_max_variance = freq_index(1) + 1;
MLR(design_parameters,s11_img_part,freq_point_with_max_variance,"img", "original")
MLR(design_parameters,s11_real_part,freq_point_with_max_variance, "real", "original")

freq_point_with_2nd_max_variance = freq_index(2) + 1;
MLR(design_parameters,s11_img_part,freq_point_with_2nd_max_variance,"img", "original")
MLR(design_parameters,s11_real_part,freq_point_with_2nd_max_variance, "real", "original")

MLR(design_parameters,s11_img_part,155+1, "img", "original")
MLR(design_parameters,s11_real_part,155+1, "real", "original")

MLR(design_parameters,s11_img_part,71+1, "img", "original")
MLR(design_parameters,s11_real_part,71+1, "real", "original")

%all local minimum for each design including first and last frequency
num_rows = size(s11_desibel, 1);
num_cols = size(s11_desibel, 2);
local_min_values_all = cell(num_rows, 1);  % Cell array to store local minima values for each row
local_min_freqs_all = cell(num_rows, 1);   % Cell array to store corresponding frequencies for each row
for row = 1:num_rows
    s11_row = s11_desibel(row, :);
    min_values = [];
    min_freqs = [];
    if s11_row(1) < s11_row(2)
        min_values = [min_values; s11_row(1)];
        min_freqs = [min_freqs; s11_freq(1)];
    end
    for col = 2:(num_cols - 1)
        if s11_row(col) < s11_row(col - 1) && s11_row(col) < s11_row(col + 1)
            min_values = [min_values; s11_row(col)]; 
            min_freqs = [min_freqs; s11_freq(col)];    
        end
    end
    if s11_row(num_cols) < s11_row(num_cols - 1)
        min_values = [min_values; s11_row(num_cols)];
        min_freqs = [min_freqs; s11_freq(num_cols)];
    end
    local_min_values_all{row} = min_values;
    local_min_freqs_all{row} = min_freqs;
end
all_min_freqs = vertcat(local_min_freqs_all{:});   
unique_freqs = unique(all_min_freqs);   
freq_counts = histc(all_min_freqs, unique_freqs); 
frequency_counts_table = table(unique_freqs, freq_counts, 'VariableNames', {'Frequency', 'Count'}); 
disp(frequency_counts_table);

unique_freqs_filtered = unique_freqs(2:end-1);  
freq_counts_filtered = freq_counts(2:end-1);  
frequency_counts_table_filtered = table(unique_freqs_filtered, freq_counts_filtered, 'VariableNames', {'Frequency Filtered', 'Count Filtered'}); 
disp(frequency_counts_table_filtered);

figure;
bar(unique_freqs_filtered, freq_counts_filtered);  
title('Frequency Distribution (Excluding First and Last Frequency)');
xlabel('Frequency');
ylabel('Count');

[max_count, idx_max] = max(freq_counts_filtered);  
most_frequent_freq = unique_freqs_filtered(idx_max); 
disp(['The most frequent frequency for the local minima (excluding first and last) is: ', num2str(most_frequent_freq)]);
disp(['It occurs ', num2str(max_count), ' times.']);

res_freq1 = most_frequent_freq+1; 
MLR(design_parameters,s11_img_part,res_freq1, "img", "original")
MLR(design_parameters,s11_real_part,res_freq1, "real", "original")

rows_to_keep = any(s11_desibel <= -10, 2);
filtered_s11_desibel_matrix = s11_desibel(rows_to_keep, :);
disp(['The number of suitable antennas: ', num2str(size(filtered_matrix,1))]);
figure;
plot(s11_freq,filtered_s11_desibel_matrix)
xlabel("Frequency")
ylabel("S11 in dB")
title("S11 in dB vs Frequency with suitable antennas")

variance_filtered_s11_desibel = var(filtered_s11_desibel_matrix);
[sorted_filtered_variance ,index_filtered] = sort(variance_filtered_s11_desibel,"descend");
freq_index_filtered = index_filtered-1;
result_table_filtered = table(sorted_variance',freq_index_filtered','VariableNames',{'Variance for Filtered Data','Frequency for Filtered Data'});
disp(result_table_filtered);

filtered_s11_img = s11_img_part(rows_to_keep,:);
filtered_design_para = design_parameters(rows_to_keep,:);
freq_point_with_max_variance_filtered = freq_index_filtered(1) + 1;
disp("For the suitable antennas:")
MLR(filtered_design_para,filtered_s11_img,freq_point_with_max_variance_filtered,"img", "original")

filtered_s11_real= s11_real_part(rows_to_keep,:);
disp("For the suitable antennas:")
MLR(filtered_design_para,filtered_s11_real,freq_point_with_max_variance_filtered,"real", "original")