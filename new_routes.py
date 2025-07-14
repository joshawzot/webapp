@app.route('/advanced-combine', methods=['POST'])
def advanced_combine():
    """Display the advanced combine input page"""
    try:
        database = request.form.get('database')
        table_names_str = request.form.get('table_names')
        
        if not database or not table_names_str:
            flash('Missing database or table information.', 'error')
            return redirect(url_for('home'))
        
        # Parse table names
        table_names = [name.strip() for name in table_names_str.split(',') if name.strip()]
        
        if len(table_names) < 2:
            flash('Advanced combine requires at least 2 tables.', 'error')
            return redirect(url_for('home'))
        
        # Get table information (dimensions and column count)
        table_info = []
        connection = create_connection()
        cursor = connection.cursor()
        
        # Check if all tables have the same number of columns
        column_counts = []
        
        for table_name in table_names:
            # Get table dimensions
            cursor.execute(f"SELECT COUNT(*) FROM `{database}`.`{table_name}`")
            row_count = cursor.fetchone()[0]
            
            # Get column count
            cursor.execute(f"SHOW COLUMNS FROM `{database}`.`{table_name}`")
            columns = cursor.fetchall()
            column_count = len(columns)
            column_counts.append(column_count)
            
            # Get first few column names to estimate dimensions
            cursor.execute(f"SHOW COLUMNS FROM `{database}`.`{table_name}` LIMIT 5")
            sample_columns = cursor.fetchall()
            
            # Try to infer dimensions from table structure
            dimensions = f"{row_count} rows, {column_count} columns"
            
            table_info.append({
                'name': table_name,
                'dimensions': dimensions,
                'row_count': row_count,
                'column_count': column_count
            })
        
        cursor.close()
        connection.close()
        
        # Check if all tables have the same number of columns
        if len(set(column_counts)) > 1:
            flash('All selected tables must have the same number of columns for advanced combine.', 'error')
            return redirect(url_for('home'))
        
        return render_template('advanced_combine_input.html', 
                             database=database, 
                             table_names=table_names_str,
                             table_info=json.dumps(table_info))
    
    except Exception as e:
        flash(f'Error preparing advanced combine: {str(e)}', 'error')
        return redirect(url_for('home'))


@app.route('/process-advanced-combine', methods=['POST'])
def process_advanced_combine():
    """Process the advanced combine operation"""
    try:
        database = request.form.get('database')
        table_names_str = request.form.get('table_names')
        new_table_name = request.form.get('new_table_name')
        num_state_values = int(request.form.get('num_state_values'))
        
        if not all([database, table_names_str, new_table_name, num_state_values]):
            flash('Missing required information.', 'error')
            return redirect(url_for('home'))
        
        # Parse table names
        table_names = [name.strip() for name in table_names_str.split(',') if name.strip()]
        
        # Get table values from form
        table_values = {}
        for i, table_name in enumerate(table_names):
            values_str = request.form.get(f'table_values_{i}')
            if not values_str:
                flash(f'Missing values for table {table_name}.', 'error')
                return redirect(url_for('home'))
            
            try:
                values = [int(v.strip()) for v in values_str.split(',')]
                if len(values) != num_state_values:
                    flash(f'Table {table_name} requires exactly {num_state_values} values.', 'error')
                    return redirect(url_for('home'))
                
                if any(v <= 0 for v in values):
                    flash(f'All values for table {table_name} must be positive integers.', 'error')
                    return redirect(url_for('home'))
                
                table_values[table_name] = values
            except ValueError:
                flash(f'Invalid values for table {table_name}. All values must be integers.', 'error')
                return redirect(url_for('home'))
        
        # Perform the advanced combine operation
        connection = create_connection()
        cursor = connection.cursor()
        
        try:
            # Get column structure from the first table
            cursor.execute(f"SHOW COLUMNS FROM `{database}`.`{table_names[0]}`")
            columns = cursor.fetchall()
            column_definitions = []
            
            for column in columns:
                column_name = column[0]
                column_type = column[1]
                column_definitions.append(f"`{column_name}` {column_type}")
            
            # Create the new table
            create_table_sql = f"""
            CREATE TABLE `{database}`.`{new_table_name}` (
                {', '.join(column_definitions)}
            )
            """
            cursor.execute(create_table_sql)
            
            # Get column names for insert statement
            column_names = [col[0] for col in columns]
            column_names_str = ', '.join([f"`{col}`" for col in column_names])
            
            # Perform the advanced combine logic
            for state_index in range(num_state_values):
                for table_name in table_names:
                    num_rows = table_values[table_name][state_index]
                    
                    # Calculate the starting row for this table and state
                    start_row = sum(table_values[table_name][:state_index])
                    
                    # Insert rows from this table
                    if num_rows > 0:
                        insert_sql = f"""
                        INSERT INTO `{database}`.`{new_table_name}` ({column_names_str})
                        SELECT {column_names_str}
                        FROM `{database}`.`{table_name}`
                        LIMIT {start_row}, {num_rows}
                        """
                        cursor.execute(insert_sql)
            
            connection.commit()
            
            # Get the final row count
            cursor.execute(f"SELECT COUNT(*) FROM `{database}`.`{new_table_name}`")
            final_row_count = cursor.fetchone()[0]
            
            cursor.close()
            connection.close()
            
            flash(f'Successfully created combined table "{new_table_name}" with {final_row_count} rows.', 'success')
            return redirect(url_for('list_tables') + f'?database={database}')
            
        except Exception as e:
            connection.rollback()
            cursor.close()
            connection.close()
            flash(f'Error during advanced combine operation: {str(e)}', 'error')
            return redirect(url_for('home'))
    
    except Exception as e:
        flash(f'Error processing advanced combine: {str(e)}', 'error')
        return redirect(url_for('home')) 