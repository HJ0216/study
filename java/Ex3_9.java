public class Ex3_9 {
    public static void main(String[] args){
        char ch = 'z';
        boolean b;

        if(ch>='A' || ch<='z') {b = true;}
        else if(ch == (ch+0)) {b = true;}
        else b = false;

        System.out.println(b);
    }
}
