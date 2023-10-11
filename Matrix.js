class Matrix {
    constructor(rows ,columns) {

        if(arguments[0] instanceof Matrix) {
            let m = arguments[0];
            this.rows = m.rows;
            this.columns = m.columns;
            this.data = new Array(this.rows);

            for(let i = 0; i < this.rows; i++) {
                this.data[i] = new Array(this.columns);
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] = m.data[i][j];
                } 
            }
        }
        else {
            this.rows = rows;
            this.columns = columns;
            this.data = new Array(this.rows);
            
            for(let i = 0; i < this.rows; i++) {
                this.data[i] = new Array(this.columns);
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] = 0;
                } 
            }
        }
    }

    randomize() {
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.columns; j++) {
                this.data[i][j] =  Math.random() * 2 - 1; 
            } 
        }
    }

    static transpose(matrix){
        let result = new Matrix(matrix.columns, matrix.rows);
        for(let i = 0; i < matrix.rows; i++) {
            for(let j = 0; j < matrix.columns; j++) {
                result.data[j][i] = matrix.data[i][j]; 
            } 
        }
        return result;
    }

    static map(matrix, func) {
        let result = new Matrix(matrix.rows, matrix.columns);
        for(let i = 0; i < result.rows; i++) {
            for(let j = 0; j < result.columns; j++) {
                let val = matrix.data[i][j];
                result.data[i][j] = func(val);
            } 
        }
        return result;
    }

    map(func) {
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.columns; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val);
            } 
        }
    }

    static fromArray(arr){
        let result = new Matrix(arr.length,1);
        for(let i = 0; i < arr.length; i++) {
            result.data[i][0] = arr[i];
        }
        return result;
    }

    toArray(){
        let array = [];
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.columns; j++) {
                array.push(this.data[i][j]);
            } 
        }
        return array;
    }

    softmax() {
        let softmaxDenominator = 0;
        for(let i = 0; i < this.rows; i++) {
            softmaxDenominator += Math.exp(this.data[i][0]);
        }
        for(let i = 0; i < this.rows; i++) {
            this.data[i][0] = ((Math.exp(this.data[i][0]) ) / softmaxDenominator);
        }
    }

    static add(a, b) {
        if(a.rows !== b.rows || a.columns !== b.columns) {
            console.log("addition is not possible!!!");
            return undefined;
        }
        let result = new Matrix(a.rows, b.columns);

        for(let i = 0; i < a.rows; i++) {
            for(let j = 0; j < b.columns; j++) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            } 
        }
        return result;
    }

    static sub(a, b) {
        if(a.rows !== b.rows || a.columns !== b.columns) {
            console.log("subtraction is not possible!!!");
            return undefined;
        }
        let result = new Matrix(a.rows, b.columns);

        for(let i = 0; i < a.rows; i++) {
            for(let j = 0; j < b.columns; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            } 
        }
        return result;
    }

    add(n) {

        if(n instanceof Matrix){
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] += n.data[i][j];
                } 
            }
        }
        else {
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] += n;
                } 
            }
        }
    }

    sub(n) {
        if(n instanceof Matrix){
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] -= n.data[i][j];
                } 
            }
        }
        else {
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] -= n;
                } 
            }
        }
    }

    static multiply(a, b){
         // Dot product
        if(a.columns != b.rows) {
            console.log("<Multiplication not possible> !!!");
            return undefined;
        }
        let result = new Matrix(a.rows, b.columns);
        
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.columns; j++) {
                for (let k = 0; k < b.rows; k++) {
                    result.data[i][j] += a.data[i][k] * b.data[k][j]; 
                } 
            }
        }
        return result;
    }

    multiply(n) {
        // Hadamard product
        if(n instanceof Matrix){
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] *= n.data[i][j];
                } 
            }
        }
        // Simple Scalar function
        else {
            for(let i = 0; i < this.rows; i++) {
                for(let j = 0; j < this.columns; j++) {
                    this.data[i][j] *= n;
                } 
            }
        }
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data){
        if(typeof data == 'string') {
            data = JSON.parse(data);
        }
        let newMatrix = new Matrix(data.rows, data.columns);
        newMatrix.data = data.data;
        return newMatrix;
    }

    print() {
        console.table(this.data);
    }
}
