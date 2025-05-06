function export_table_summary(filename,data,row_names,col_names)
% Purpose: Produce latex table for given data and row and column names

% Dimensions
N=length(row_names);
K=length(col_names);

% number format
format='%1.2f\n';
if isnumeric(data)
   data_names=num2cell(data);
    for n=1:size(data,1) %#ok<ALIGN>
        for k=1:size(data,2)
            data_names{n,k}=num2str(data_names{n,k},format);
        end
    end
end
C='X';
for i=1:K/2
    C=[C,'@{\\hspace{0.25cm}}c'];
end

fid=fopen(filename, 'w');
fprintf(fid,'\\begin{table}[h!]\r\n');
fprintf(fid,'\\fontsize{11}{18}\\selectfont{\r\n');
fprintf(fid,'\\begin{center}\r\n');
fprintf(fid,'\\caption{Descriptive statistics on the returns, target prices and recommendations for 20 Dow Jones constituents (sample: 1999 - 2015) }\r\n');
fprintf(fid,'\\label{tab:summary}\r\n');
fprintf(fid,'\\vspace{-0.2cm}\r\n');
fprintf(fid,['\\begin{tabularx}{1\\textwidth}{@{}',C,'@{}}\r\n']);
fprintf(fid,'\\toprule\\toprule\r\n');

for i=1:2*N
    if i==1 
        fprintf(fid, ' %s ', row_names{i});
        for j=1:K/2
            fprintf(fid, ' & %s	', col_names{j});
        end
        fprintf(fid,'\\\\\r\n');
        fprintf(fid,'\\midrule\r\n');
    elseif i>1 && i<N+1
        fprintf(fid, ' %s ', row_names{i});
        for j=1:K/2
            fprintf(fid, ' & %s	',num2str(data_names{i-1,j}));
        end
        fprintf(fid,'\\\\\r\n');
        if i==3 || i==6
            fprintf(fid,'\\midrule\r\n');
        end
    elseif i==N+1
        fprintf(fid,'\\midrule\r\n');
        fprintf(fid,'\\midrule\r\n');
        fprintf(fid, ' %s ', row_names{i-N});
        for j=1:K/2
            fprintf(fid, ' & %s	', col_names{j+K/2});
        end
        fprintf(fid,'\\\\\r\n');
        fprintf(fid,'\\midrule\r\n');
    elseif i>N+1
        fprintf(fid, ' %s ', row_names{i-N});
        for j=1:K/2
            fprintf(fid, ' & %s	',num2str(data_names{i-N-1,j+K/2}));
        end
        fprintf(fid,'\\\\\r\n');
        if i==3+N || i==6+N
            fprintf(fid,'\\midrule\r\n');
        end
    end
end

fprintf(fid,'\\bottomrule\\bottomrule\r\n');
fprintf(fid,'\\end{tabularx}\r\n');
fprintf(fid,'\\vspace{0.2cm}\r\n');
fprintf(fid,'\\caption*{\\footnotesize \\textit{Note:} The table reports descriptive statistics on the returns, expected target returns and recommendations  for 20 Dow Jones constituents. It reports the mean and standard deviation of the logarithmic monthly returns, the mean number of available target prices, the mean and variance of the monthly forward target price implied expected return, i.e. simple returns between the spot and the twelve month forward target price at each point $t$ divided by 12, constructed from individual analyst data, the number of recommendations as well as the mean and standard deviation of the recommendations based on the 1 (strong buy) to 5 (strong sell) scale. Mean returns and standard deviations are multiplied by 100. Target prices and recommendations are obtained from I/B/E/S Datastream.}\r\n');
fprintf(fid,'\\end{center}}\r\n');
fprintf(fid,'\\end{table}');
fclose(fid);
end



% old

% % number format
% format='%7.3f\n';
% if isnumeric(data)
%    data_names=num2cell(data);
%     for n=1:N-1 %#ok<ALIGN>
%         for k=1:K-1
% 
%             if k==1 && data(n,1)<data(n,2)
%                 data_names{n,k}=['\textbf{',num2str(data_names{n,k},format),'}'];
%             elseif k==1 && data(n,1)>data(n,2)
%                 data_names{n,k}=num2str(data_names{n,k},format);
%             elseif k==2 && data(n,1)<data(n,2)
%                 data_names{n,k}=num2str(data_names{n,k},format);
%             elseif k==2 && data(n,1)>data(n,2)
%                 data_names{n,k}=['\textbf{',num2str(data_names{n,k},format),'}'];
%             end
%             
%         end
%     end
% end
% 
% % C='X';
% % for i=1:K-1
% %     C=[C,'r'];
% % end
% 
% C=['X','r','r','@{\\hspace{0.7cm}}l','r','r'];
% 
% fid=fopen(filename, 'w');
% fprintf(fid,'\\begin{table}[h!]\r\n');
% fprintf(fid,'\\fontsize{11}{18}\\selectfont{\r\n');
% fprintf(fid,'\\begin{center}\r\n');
% fprintf(fid,'\\caption{Root mean squared errors between forecasted and observed spot prices for 22 Dow Jones constituents (sample: 1999 - 2015) }\r\n');
% fprintf(fid,'\\label{tab:}\r\n');
% fprintf(fid,['\\begin{tabularx}{0.6\\textwidth}{@{}',C,'@{}}\r\n']);
% fprintf(fid,'\\toprule\\toprule\r\n');
% 
% for i=1:(N-1)/2
%     if i==1
%         fprintf(fid, ' %s ', col_names{1});
%         for j=2:K
%             fprintf(fid, ' & %s	', col_names{j});
%         end
%         fprintf(fid, ' & %s ', col_names{1});
%         for j=2:K
%             fprintf(fid, ' & %s	', col_names{j});
%         end
%         fprintf(fid,'\\\\\r\n');
%         fprintf(fid,'\\midrule\r\n');
%     else
%         fprintf(fid, ' %s ', row_names{i-1});
%         for j=2:K
%             fprintf(fid, ' & %s	',num2str(data_names{i-1,j-1}));
%         end
%         fprintf(fid, ' & %s ', row_names{i+(N-1)/2});
%         for j=2:K
%             fprintf(fid, ' & %s	',num2str(data_names{i+(N-1)/2,j-1}));
%         end
%         fprintf(fid,'\\\\\r\n');
%     end
% end
% 
% fprintf(fid,'\\bottomrule\\bottomrule\r\n');
% fprintf(fid,'\\end{tabularx}\r\n');
% fprintf(fid,'\\vspace{0.2cm}\r\n');
% fprintf(fid,'\\caption*{\\footnotesize \\textit{Note:} The table displays root mean squared errors between observed spot price twelve months ahead and mean 12-month forward target price and two year historical average  for 22 Dow Jones constituents between 1999 and 2015.}\r\n');
% fprintf(fid,'\\end{center}}\r\n');
% fprintf(fid,'\\end{table}');
% fclose(fid);
% end

