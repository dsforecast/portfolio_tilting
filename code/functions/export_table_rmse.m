function export_table_rmse(filename,data,tests,row_names,col_names)
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
            if data(n,k)>=1
                if tests(n,k)>0.1
                    data_names{n,k}=num2str(data_names{n,k},format);     
                elseif tests(n,k)<=0.1 && tests(n,k)>0.05
                     data_names{n,k}=['$',num2str(data_names{n,k},format),'^{*}$'];
                elseif tests(n,k)<=0.05 && tests(n,k)>0.01
                     data_names{n,k}=['$',num2str(data_names{n,k},format),'^{**}$'];
                elseif tests(n,k)<=0.01
                     data_names{n,k}=['$',num2str(data_names{n,k},format),'^{***}$'];
                end
            else
                if tests(n,k)>0.10
                    data_names{n,k}=['\textbf{',num2str(data_names{n,k},format),'}'];     
                elseif tests(n,k)<=0.1 && tests(n,k)>0.05               
                     data_names{n,k}=['$\mathbf{',num2str(data_names{n,k},format),'^{*}}$'];
                elseif tests(n,k)<=0.05 && tests(n,k)>0.01
                     data_names{n,k}=['$\mathbf{',num2str(data_names{n,k},format),'^{**}}$'];
                elseif tests(n,k)<=0.01
                     data_names{n,k}=['$\mathbf{',num2str(data_names{n,k},format),'^{***}}$'];
                end
            end
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
fprintf(fid,'\\caption{Relative root mean squared errors between forecasted and observed spot prices for 20 Dow Jones constituents (sample: 1999 - 2015) }\r\n');
fprintf(fid,'\\label{tab:rmsfe}\r\n');
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
    elseif i==2
        fprintf(fid, ' %s ', row_names{2});
        for j=1:K/2
            fprintf(fid, ' & %s	',num2str(data_names{j,1}));
        end
        fprintf(fid,'\\\\\r\n');
        fprintf(fid,'\\midrule\r\n');
    elseif i==3
        fprintf(fid,'\\midrule\r\n');
        fprintf(fid, ' %s ', row_names{1});
        for j=1:K/2
            fprintf(fid, ' & %s	', col_names{j+K/2});
        end
        fprintf(fid,'\\\\\r\n');
        fprintf(fid,'\\midrule\r\n');
    elseif i==4
        fprintf(fid, ' %s ', row_names{2});
        for j=1:K/2
            fprintf(fid, ' & %s	',num2str(data_names{j+K/2,1}));
        end
        fprintf(fid,'\\\\\r\n');
    end
end

fprintf(fid,'\\bottomrule\\bottomrule\r\n');
fprintf(fid,'\\end{tabularx}\r\n');
fprintf(fid,'\\vspace{0.2cm}\r\n');
fprintf(fid,'\\caption*{\\footnotesize \\textit{Note:} The table displays relative root mean squared errors between observed spot price twelve months ahead and the mean 12-month forward target price as well as the two year historical average for 20 Dow Jones constituents between 1999 and 2015. Values lower than one indicate that the target price generates superior forecast performance. For each stock, we test whether the target price forecast has lower MSFE than the average price forecast by the test proposed by \\citet{giacomini2006}. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.}\r\n');
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

